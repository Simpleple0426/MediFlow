"""医疗咨询字段准备和共享知识检索工具；不执行诊断、处方或实际预约。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler
    allowed_agents: Optional[Tuple[str, ...]] = None
    retry_safe: bool = True


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
    *,
    allowed_agents: Optional[Tuple[str, ...]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
        allowed_agents=allowed_agents,
    )


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = args.get("top_k", 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "cached": bool(getattr(result, "cached", False)),
            "reranked": bool(getattr(result, "reranked", False)),
        }

    from agents.domain_tools import build_domain_search_tools
    return {
        **build_domain_search_tools(search_knowledge_base),
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于健康接待、分诊、用药和挂号指引。",
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000, "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def consultation_plan(intent: str) -> Dict[str, Any]:
    """细意图决定咨询重点与补充字段；清单不执行医学核验。"""
    plans = {
        'symptom_query': ('症状信息整理与就医方向', ['年龄', '主要症状', '发生时间及持续时间', '体温和伴随症状']),
        'medication_query': ('药品信息与说明书核对', ['药品通用名、规格及说明书', '年龄', '过敏史', '孕哺期及肝肾情况']),
        'medication_interaction': ('联合用药信息核对，不宣称已验证安全', ['全部合并用药通用名及规格', '各药用法及服用时间', '年龄', '过敏史', '孕哺期及肝肾情况']),
        'medication_adverse_reaction': ('疑似不良反应信息整理，不确定因果或建议自行停换药', ['药品通用名及规格', '开始用药时间', '不适表现及出现时间', '严重程度和危险信号', '全部合并用药', '过敏史']),
        'medication_dosage': ('解释已有用法用量信息，不制定或调整个人剂量', ['药品通用名、剂型及规格', '原处方或说明书用法', '实际服用量和时间', '年龄及体重（儿童）', '孕哺期及肝肾情况']),
        'report_query': ('解释报告字段含义，不作诊断', ['脱敏指标名称', '数值、单位及参考范围', '检查日期', '年龄及相关症状']),
        'report_abnormal': ('整理异常指标与就医核对问题，单项异常不能确诊', ['脱敏异常指标及数值', '单位及本机构参考范围', '检查日期和既往结果', '相关症状及病史']),
        'appointment': ('官方挂号流程及材料准备，不查号源或实际预约', ['目标医院（可选）', '就诊方向', '初诊或复诊']),
        'department_query': ('提供科室选择方向并提示医院导诊核对', ['主要症状或咨询问题', '年龄', '持续时间', '已有诊断或转诊建议（如有）']),
        'greeting': ('介绍医疗咨询范围', []),
        'human_handoff': ('说明人工联系渠道，不声称已接通', []),
        'emergency': ('优先急救提示，不等待字段补齐', []),
        'other': ('澄清医疗咨询需求', ['咨询方向']),
    }
    focus, fields = plans.get(intent, plans['other'])
    return {'consultation_focus': focus, 'fields_to_confirm': fields}


def medical_tools() -> Dict[str, AgentToolSpec]:
    from agents.domain_tools import build_domain_tools
    def prepare(req, args):
        intent = req.intent.value if req.intent else ''
        return {'success': True, 'context_available': bool(req.context), **consultation_plan(intent),
                'note': '从本轮原话与上下文核对清单，只追问尚缺的信息；历史画像需核对时效。清单齐全不代表诊断或处方核验完成。',
                'diagnosis_provided': False, 'appointment_created': False}
    return {
        'prepare_health_consultation': make_tool('prepare_health_consultation',
            '整理脱敏咨询字段与需向用户确认的信息，不诊断、不查号源、不核验药物相互作用。', {}, prepare),
        **build_consultation_tools(),
        **build_domain_tools(),
    }


def build_consultation_tools() -> Dict[str, AgentToolSpec]:
    """无外部依赖的资料整理工具；私有权限由 allowed_agents 声明。"""
    text = {"type": "string", "minLength": 1, "maxLength": 500, "pattern": r"\S"}
    texts = {"type": "array", "items": text, "maxItems": 30}

    def summary(req, args):
        return {"success": True, "summary": {
            "chief_concern": args["chief_concern"],
            "known_facts": args.get("known_facts", []),
            "questions_for_clinician": args.get("questions", []),
        }, "source": "user_provided", "persisted": False,
            "note": "仅整理用户提供并确认的信息，不推断病史，不代表医生病历或已完成转接。"}

    def compare_reports(req, args):
        results = []
        for item in args["items"]:
            value, low, high = item["value"], item.get("reference_low"), item.get("reference_high")
            if any(not math.isfinite(x) for x in (value, low, high) if x is not None):
                return {"success": False, "error": "报告数值和范围必须是有限数字", "results": []}
            if low is not None and high is not None and low > high:
                return {"success": False, "error": "参考范围下限不能大于上限", "results": []}
            if low is None and high is None:
                comparison = "unknown"
            elif low is not None and value < low:
                comparison = "below"
            elif high is not None and value > high:
                comparison = "above"
            elif low is None or high is None:
                comparison = "within_supplied_bound"
            else:
                comparison = "within"
            results.append({**item, "comparison": comparison})
        return {"success": True, "results": results, "diagnosis_provided": False,
            "note": "仅按报告原文所给范围做数值比较；数值和上下限必须使用同一单位，不推断缺失界限，不作临床正常或异常诊断。"}

    def medications(req, args):
        fields = ("name", "strength", "actual_usage", "started_at")
        entries = [{**item, "missing_fields": [key for key in fields if not item.get(key)]}
                   for item in args["medications"]]
        return {"success": True, "medications": entries,
            "allergies": args.get("allergies", []), "allergy_history_confirmed": "allergies" in args,
            "interaction_checked": False, "dose_recommended": False,
            "note": "保留用户原有药名及实际用法，缺失项待核对；不合并同名药，不推断成分，不验证合用安全，不调整剂量。"}

    def visit(req, args):
        visit_type = args["visit_type"]
        checklist = ["在目标医院官方渠道核对科室、挂号流程及材料要求",
                     "按医院要求准备就诊凭证，勿在聊天中提供证件号码",
                     "整理本次咨询问题、已有检查报告及用药清单"]
        if visit_type == "follow_up":
            checklist.append("整理既往就诊记录和医生交代的复诊事项")
        return {"success": True, "hospital": args.get("hospital"),
            "department": args.get("department"), "visit_type": visit_type,
            "checklist": checklist, "questions_for_hospital": args.get("questions", []),
            "appointment_created": False, "availability_checked": False,
            "note": "通用准备清单，具体要求以医院官方信息为准；未查询实时流程、号源或完成挂号。"}

    report_item = {"type": "object", "properties": {
        "name": text, "value": {"type": "number"}, "unit": text,
        "reference_low": {"type": "number"}, "reference_high": {"type": "number"},
    }, "required": ["name", "value", "unit"], "additionalProperties": False}
    medication_item = {"type": "object", "properties": {
        key: text for key in ("name", "strength", "actual_usage", "started_at")
    }, "required": ["name"], "additionalProperties": False}
    return {
        "build_visit_summary": make_tool("build_visit_summary",
            "将已确认的脱敏事实和待问问题整理为就诊摘要；不自动读取或保存患者档案。",
            {"chief_concern": text, "known_facts": texts, "questions": texts}, summary,
            required=["chief_concern"]),
        "compare_report_ranges": make_tool("compare_report_ranges",
            "分诊私有：仅比较报告原文数值与用户提供的同单位参考上下限；不得补造范围或据此诊断。",
            {"items": {"type": "array", "items": report_item, "minItems": 1, "maxItems": 30}},
            compare_reports, required=["items"], allowed_agents=("triage",)),
        "organize_medication_list": make_tool("organize_medication_list",
            "用药咨询私有：整理用户已服用药品和实际用法，标出缺失字段；不验证相互作用或推荐剂量。",
            {"medications": {"type": "array", "items": medication_item, "minItems": 1, "maxItems": 30},
             "allergies": texts}, medications, required=["medications"], allowed_agents=("medication",)),
        "prepare_visit_checklist": make_tool("prepare_visit_checklist",
            "挂号导诊私有：按初诊或复诊生成通用就诊准备清单；不查号源、不生成预约。",
            {"visit_type": {"type": "string", "enum": ["first_visit", "follow_up", "unknown"]},
             "hospital": text, "department": text, "questions": texts},
            visit, required=["visit_type"], allowed_agents=("appointment",)),
    }
