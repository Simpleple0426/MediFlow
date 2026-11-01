"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import math
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from core.llm_client import create_client

from core.llm_utils import create_message, extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    SYMPTOM_QUERY = "symptom_query"
    MEDICATION_QUERY = "medication_query"
    MEDICATION_INTERACTION = "medication_interaction"
    MEDICATION_ADVERSE_REACTION = "medication_adverse_reaction"
    MEDICATION_DOSAGE = "medication_dosage"
    APPOINTMENT = "appointment"
    DEPARTMENT_QUERY = "department_query"
    REPORT_QUERY = "report_query"
    REPORT_ABNORMAL = "report_abnormal"
    EMERGENCY = "emergency"
    GREETING = "greeting"
    HUMAN_HANDOFF = "human_handoff"
    OTHER = "other"


class IntentGroup(Enum):
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    REPORT = "report"
    APPOINTMENT = "appointment"
    EMERGENCY = "emergency"
    GENERAL = "general"


_INTENT_GROUPS = {
    IntentCategory.SYMPTOM_QUERY: IntentGroup.SYMPTOM,
    IntentCategory.MEDICATION_QUERY: IntentGroup.MEDICATION,
    IntentCategory.MEDICATION_INTERACTION: IntentGroup.MEDICATION,
    IntentCategory.MEDICATION_ADVERSE_REACTION: IntentGroup.MEDICATION,
    IntentCategory.MEDICATION_DOSAGE: IntentGroup.MEDICATION,
    IntentCategory.REPORT_QUERY: IntentGroup.REPORT,
    IntentCategory.REPORT_ABNORMAL: IntentGroup.REPORT,
    IntentCategory.APPOINTMENT: IntentGroup.APPOINTMENT,
    IntentCategory.DEPARTMENT_QUERY: IntentGroup.APPOINTMENT,
    IntentCategory.EMERGENCY: IntentGroup.EMERGENCY,
    IntentCategory.GREETING: IntentGroup.GENERAL,
    IntentCategory.HUMAN_HANDOFF: IntentGroup.GENERAL,
    IntentCategory.OTHER: IntentGroup.GENERAL,
}
_GROUP_DEFAULTS = {
    IntentGroup.SYMPTOM: IntentCategory.SYMPTOM_QUERY,
    IntentGroup.MEDICATION: IntentCategory.MEDICATION_QUERY,
    IntentGroup.REPORT: IntentCategory.REPORT_QUERY,
    IntentGroup.APPOINTMENT: IntentCategory.APPOINTMENT,
    IntentGroup.EMERGENCY: IntentCategory.EMERGENCY,
    IntentGroup.GENERAL: IntentCategory.OTHER,
}


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 兼容字段；画像在后台独立提取
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    embedding_source: str = "local-ngram"


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.SYMPTOM_QUERY: ["我咳嗽三天了，需要补充哪些信息？", "发烧头痛应该如何就医？"],
    IntentCategory.MEDICATION_QUERY: ["药品说明书里的禁忌怎么看？", "药品通用名是什么意思？"],
    IntentCategory.MEDICATION_INTERACTION: ["这两种药可以一起吃吗？", "药物相互作用怎么核对？"],
    IntentCategory.MEDICATION_ADVERSE_REACTION: ["服药后出现皮疹，是不良反应吗？", "吃药后恶心需要向药师说明什么？"],
    IntentCategory.MEDICATION_DOSAGE: ["药品说明书的用法用量怎么看？", "这药一次吃几片？"],
    IntentCategory.APPOINTMENT: ["第一次去医院怎么挂号？", "就诊需要带哪些材料？"],
    IntentCategory.DEPARTMENT_QUERY: ["头痛应该挂哪个科？", "怎么选择就诊科室？"],
    IntentCategory.REPORT_QUERY: ["检查报告上的参考范围是什么意思？", "血常规指标名称是什么意思？"],
    IntentCategory.REPORT_ABNORMAL: ["血常规指标偏高怎么办？", "报告上的白细胞低就能确诊吗？"],
    IntentCategory.EMERGENCY: ["我现在胸痛而且呼吸困难", "有人突然意识丧失叫不醒"],
    IntentCategory.GREETING: ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.HUMAN_HANDOFF: ["我想联系人工医务人员", "请帮我联系医生", "找人工协助"],
    IntentCategory.OTHER: ["聊聊电影", "这个问题不属于健康咨询"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    LLM 使用项目统一客户端，向量使用独立配置的 Embedding 客户端。
    模板向量在首次请求时生成并缓存；在线不可用时整次匹配回退本地 n-gram。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = create_client(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        from core.intent_embedding import IntentEmbedding
        self._embedding = IntentEmbedding()
        embedding_enabled = os.getenv('INTENT_EMBEDDING_ENABLED', 'true').strip().lower()
        if embedding_enabled not in {'true', 'false', '1', '0', 'yes', 'no', 'on', 'off'}:
            raise ValueError('INTENT_EMBEDDING_ENABLED 必须为 true/false')
        self._embedding_enabled = embedding_enabled in {'true', '1', 'yes', 'on'}
        self._cache: Dict[str, IntentResult] = {}
        self._cache_expiries: Dict[str, float] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        from core.healthcare import healthcare_enabled, emergency_signals
        if healthcare_enabled() and emergency_signals(message):
            return IntentResult(IntentCategory.EMERGENCY, 1.0, UrgencyLevel.CRITICAL,
                "emergency", {}, "命中急症危险信号，跳过模型与在线向量", 0,
                {"safety_rule": 1.0}, "disabled")
        key = self._cache_key(message, history)
        if key in self._cache and time.monotonic() < self._cache_expiries.get(key, 0):
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和向量分支并行；provider=local 仍参与向量投票。
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0, "source": "disabled"}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        if healthcare_enabled() and intent == IntentCategory.EMERGENCY and not emergency_signals(message):
            intent = IntentCategory.SYMPTOM_QUERY
        entities = {}
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
            embedding_source=emb.get("source", "unavailable"),
        )

        # 进程内识别结果缓存：写入前达到容量时，按插入顺序批量淘汰。
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
                self._cache_expiries.pop(k, None)
        self._cache[key] = result
        self._cache_expiries[key] = time.monotonic() + (60 if result.embedding_source == "local-ngram" and self._embedding.enabled else 300)
        return result

    async def check_embedding(self, message: str) -> Dict[str, Any]:
        """独立验证向量链路；不调用 LLM，也不写入意图识别结果缓存。"""
        if not self._embedding_enabled:
            return {'success': False, 'source': 'disabled', 'error': 'embedding_branch_disabled'}
        started = time.monotonic()
        result = await self._embedding_recognize(message)
        source = result.get('source', 'unavailable')
        return {
            'success': source == 'siliconflow/' + self._embedding.MODEL,
            'source': source,
            'intent': result['intent'].value,
            'confidence': result['confidence'],
            'elapsed_ms': round((time.monotonic() - started) * 1000, 1),
            'status': self._embedding.status(),
        }

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        if not isinstance(correct, IntentCategory):
            raise ValueError("纠正意图必须属于当前13个医疗咨询细意图")
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._embedding.remote_templates.clear()
            self._embedding.local_templates.clear()
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            self._cache_expiries.clear()
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 条历史消息（按消息条数计）
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是医疗健康咨询意图分析助手。优先从下列13个细意图中选择，返回 JSON。
六个粗分类为 symptom、medication、report、appointment、emergency、general。
用药：一般药品信息 medication_query；联合用药/相互作用 medication_interaction；
疑似不良反应 medication_adverse_reaction；用法用量 medication_dosage。
报告：字段含义 report_query；指标异常 report_abnormal。
就诊：挂号流程 appointment；科室选择 department_query。
若仅能确定领域但无法确定细意图，可返回粗分类值；不要猜测不存在的症状。
非医疗业务问题归为 other；明确希望联系医务人员或人工协助归为 human_handoff。
否认症状、历史科普或假设问题不代表当前正在发生急症。

示例:
{examples}

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await asyncio.wait_for(create_message(self.client,
                stage="intent",
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            ), timeout=max(0.1, float(os.getenv("MEDIFLOW_INTENT_TIMEOUT_S", "8"))))
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                try:
                    data["intent"] = IntentGroup(data["intent"])
                except ValueError:
                    data["intent"] = IntentCategory.OTHER
            confidence = data.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("意图 confidence 必须是 0 到 1 的有限数字")
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            templates, msg_vec, source = await self._embedding.vectors(
                self._clean_text(message), _TEMPLATES, self._local_embedding)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in templates.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": min(1.0, max(0.0, best_score)), "source": source}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "source": "unavailable"}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """医疗规则兜底；人工请求先于普通咨询，急症在 recognize 入口处理。"""
        from core.healthcare import medical_rule_intent
        msg = message.lower()
        if any(term in msg for term in ("转人工", "找人工", "人工协助", "联系医生", "联系药师", "联系人工", "human handoff")):
            return {"intent": IntentCategory.HUMAN_HANDOFF, "confidence": 0.95}
        medical = medical_rule_intent(message)
        if medical:
            return {"intent": IntentCategory(medical), "confidence": 0.9}
        if re.fullmatch(r"(?:你好|您好|嗨|早上好|晚上好|hello|hi)[！!。,.，\s]*", msg):
            return {"intent": IntentCategory.GREETING, "confidence": 0.9}
        return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """先聚合粗分类得分，再选择组内细意图；仅允许同组规则细化。"""
        cleaned = []
        for source in (llm, emb, pat):
            value = source.get("confidence")
            if (not isinstance(source.get("intent"), (IntentCategory, IntentGroup)) or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
                source = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
            cleaned.append(source)
        llm, emb, pat = cleaned
        from core.healthcare import healthcare_enabled
        medical_intents = set(IntentCategory) - {IntentCategory.OTHER, IntentCategory.EMERGENCY}
        if healthcare_enabled() and (llm.get('failed') or llm.get('intent') == IntentCategory.OTHER) and pat.get('intent') in medical_intents:
            return pat['intent'], float(pat['confidence']), {'llm': 0.0,
                'embedding': float(emb.get('confidence', 0)), 'pattern': float(pat['confidence'])}
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return self._fine_intent(emb["intent"]), source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return self._fine_intent(pat["intent"]), source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[Any, float] = {}
        group_scores: Dict[str, float] = {}
        for result, weight in weights:
            cat = result["intent"]
            scores[cat] = scores.get(cat, 0.0) + weight * result["confidence"]
            group = self._intent_group(cat)
            group_scores[group] = group_scores.get(group, 0.0) + weight * result["confidence"]
        winning_group = max(group_scores, key=group_scores.get)
        candidates = {cat: score for cat, score in scores.items() if self._intent_group(cat) == winning_group}
        best = max(candidates, key=candidates.get)
        best_score = group_scores[winning_group]
        source_scores.update({"group_" + group: score for group, score in group_scores.items()})
        # 延续原项目：粗候选或组内通用意图可被同组明确细规则细化。
        # 已识别的具体子意图不能被另一个子意图覆盖；通用组不做该细化。
        refinable = isinstance(best, IntentGroup) or best in {
            IntentCategory.MEDICATION_QUERY, IntentCategory.REPORT_QUERY, IntentCategory.APPOINTMENT}
        if (refinable and isinstance(pat["intent"], IntentCategory)
                and pat["intent"] != self._fine_intent(best)
                and self._intent_group(pat["intent"]) == winning_group
                and winning_group != IntentGroup.GENERAL.value
                and pat["confidence"] >= 0.5 and best_score < 0.8):
            best = pat["intent"]
            best_score = max(best_score, pat["confidence"])
            source_scores["refined_by_pattern"] = 1.0
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return self._fine_intent(best), best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """Compatibility field: profile extraction runs only in memory maintenance."""
        return {}

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        from core.healthcare import emergency_signals, high_fever
        if intent == IntentCategory.EMERGENCY or emergency_signals(message):
            return UrgencyLevel.CRITICAL
        if high_fever(message) or intent == IntentCategory.HUMAN_HANDOFF:
            return UrgencyLevel.HIGH
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", "")),
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory | IntentGroup) -> str:
        return (intent if isinstance(intent, IntentGroup) else _INTENT_GROUPS[intent]).value

    @staticmethod
    def _fine_intent(intent: IntentCategory | IntentGroup) -> IntentCategory:
        return _GROUP_DEFAULTS[intent] if isinstance(intent, IntentGroup) else intent

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
