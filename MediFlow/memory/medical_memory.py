"""Patient-scoped durable conversation memory, separate from shared RAG documents."""
import asyncio
import json
import logging
from datetime import datetime

from core.concurrency import redis_lease
from core.patient_identity import (PatientScopeConflict, check_patient_label, digest, memory_scope)
from core.medical_entities import extract_profile_patch
from core.patient_profile import apply_patch, sanitize_profile

logger = logging.getLogger(__name__)


class MedicalMemory:
    """Mixin used by MemoryManager; all Chroma I/O stays outside the event loop."""

    async def _read_medical_state(self, user_id, conv_id):
        raw = await self._redis.get(self._state_key(user_id, conv_id))
        return json.loads(raw) if raw else {}

    @staticmethod
    def _message_in_scope(message, scope):
        saved = message.metadata.get("memory_scope", {})
        # Legacy messages have no reliable patient identity; use only in private session scope.
        return saved.get("key") == scope["key"] or (
            not saved and scope["source"] == "session" and not scope.get("epoch")
            and scope["label"] == "未确认咨询对象")

    async def _medical_context(self, user_id, conv_id, query, patient_id=None):
        from memory.conversation_memory import MemoryContext
        state = await self._read_medical_state(user_id, conv_id)
        scope = memory_scope(conv_id, query, state, patient_id)
        # Resolve legacy identity first, then discard all old clinical slots.
        state = {"domain": "healthcare", "memory_scope": scope}
        recent = [m for m in await self._get_working_memory(user_id, conv_id)
                  if self._message_in_scope(m, scope)]
        summary = await self._redis.get(self._medical_summary_key(user_id, conv_id, scope)) or ""
        history, profile = await asyncio.gather(
            self._medical_search(user_id, conv_id, query, scope), self._get_profile(user_id, scope))
        if scope["source"] == "explicit" and profile:
            check_patient_label(profile.get("patient_label"), scope["label"])
            if scope["label"] == "指定患者":
                scope["label"] = profile.get("patient_label", scope["label"])
        return MemoryContext(recent, history, profile, summary, state)

    def _medical_summary_key(self, user_id, conv_id, scope):
        base = self._summary_key(user_id, conv_id)
        legacy = scope["source"] == "session" and not scope.get("epoch") and scope["label"] == "未确认咨询对象"
        return base if legacy else base + ":" + digest(scope["key"])

    async def _medical_search(self, user_id, conv_id, query, scope):
        if not query.strip():
            return []
        try:
            results = await self._query_episodic(query, self.HISTORY_TOP_K,
                {"$and": [{"user_id": user_id}, {"patient_key": scope["key"]}, {"domain": "healthcare"}]})
            return self._dedupe_texts(self._extract_docs(results))[:self.HISTORY_TOP_K]
        except Exception as ex:
            logger.warning("长期医疗记忆检索失败: %s", type(ex).__name__)
            return []

    async def _medical_profile(self, user_id, scope, *, strict=False):
        if not scope:
            return {}
        try:
            result = await asyncio.to_thread(self._profile.get, ids=["patient_profile:" + digest(user_id, scope["key"])])
            for document, meta in zip(result.get("documents") or [], result.get("metadatas") or []):
                if meta.get("domain") == "healthcare" and meta.get("user_id") == user_id and meta.get("patient_key") == scope["key"]:
                    value = json.loads(document)
                    return sanitize_profile(value, value.get("patient_label", scope["label"])) if isinstance(value, dict) else {}
        except Exception as ex:
            logger.warning("长期医疗画像读取失败: %s", type(ex).__name__)
            if strict:
                raise
        return {}

    def _medical_turns(self, messages, conv_id):
        turns = {}
        for msg in messages:
            request_id = msg.metadata.get("request_id")
            scope = msg.metadata.get("memory_scope")
            if not request_id or not isinstance(scope, dict) or not scope.get("key"):
                continue
            key = (request_id, scope["key"])
            row = turns.setdefault(key, {"scope": scope, "request_id": request_id, "conv_id": conv_id})
            row[msg.role.value] = msg
        return [row for row in turns.values() if "user" in row and "assistant" in row]

    async def _archive_medical_messages(self, user_id, conv_id, messages):
        turns = self._medical_turns(messages, conv_id)
        if not turns:
            return True
        ids, documents, metadata = [], [], []
        for row in turns:
            scope, user = row["scope"], row["user"]
            stamp = user.timestamp.isoformat()
            ids.append("turn:" + digest(user_id, conv_id, scope["key"], row["request_id"]))
            documents.append(self._safe_text(
                f"[历史咨询记录，需确认是否仍适用，不代表当前病情或已核验事实]\n"
                f"记录时间：{stamp}；咨询对象：{scope['label']}\n"
                f"用户陈述：{user.content}\n助手当时回复：{row['assistant'].content}"))
            metadata.append({"domain": "healthcare", "user_id": user_id,
                "patient_key": scope["key"], "patient_label": scope["label"],
                "conv_id": conv_id, "request_id": row["request_id"], "ts": stamp, "kind": "turn"})
        try:
            await asyncio.wait_for(asyncio.to_thread(self._episodic.upsert,
                ids=ids, documents=documents, metadatas=metadata), timeout=10)
            return True
        except Exception as ex:
            logger.warning("长期医疗对话归档失败，保留近期原文等待后续维护: %s", type(ex).__name__)
            return False

    async def _store_medical_summary(self, user_id, conv_id, text, summary, scope):
        stamp = datetime.now().isoformat()
        await asyncio.to_thread(self._episodic.upsert,
            ids=["summary:" + digest(user_id, conv_id, scope["key"], text)],
            documents=[f"[历史咨询摘要，需确认是否仍适用]\n对象：{scope['label']}；记录时间：{stamp}\n{summary}"],
            metadatas=[{"domain": "healthcare", "user_id": user_id, "conv_id": conv_id,
                "patient_key": scope["key"], "patient_label": scope["label"], "ts": stamp, "kind": "summary"}])

    async def _maintain_medical(self, user_id, conv_id):
        # Coalesce one conversation, never discard another patient's pending conversation.
        async with redis_lease(self._redis, "lock:memory-maintain:" + digest(user_id, conv_id)):
            messages = self._messages_from_raw(await self._redis.lrange(self._wm_key(user_id, conv_id), 0, -1))
            if not await self._archive_medical_messages(user_id, conv_id, messages):
                return
            await self.update_profile(user_id, conv_id)
            await asyncio.wait_for(self._compress(user_id, conv_id), 30)

    async def _update_medical_profiles(self, user_id, conv_id):
        messages = self._messages_from_raw(await self._redis.lrange(self._wm_key(user_id, conv_id), 0, -1))
        grouped = {}
        for turn in self._medical_turns(messages, conv_id):
            grouped.setdefault(turn["scope"]["key"], []).append(turn)
        for patient_key, turns in grouped.items():
            scope = turns[-1]["scope"]
            async with redis_lease(self._redis, "lock:medical-profile:" + digest(user_id, patient_key)):
                profile = await self._medical_profile(user_id, scope, strict=True)
                if profile:
                    check_patient_label(profile.get("patient_label"), scope["label"])
                    if scope["label"] == "指定患者":
                        scope = {**scope, "label": profile["patient_label"]}
                if scope["label"] == "多位患者待确认":
                    continue
                processed = []
                changed = False
                for turn in sorted(turns, key=lambda t: t["user"].timestamp):
                    user = turn["user"]
                    marker = "profile-patch:v2:" + digest(user_id, conv_id, patient_key, turn["request_id"])
                    if await self._redis.get(marker):
                        continue
                    patch = await extract_profile_patch(user.content, scope["label"], self._client, self._model)
                    if any(patch.values()):
                        profile = apply_patch(profile, patch, scope, user.timestamp.isoformat(), conv_id, turn["request_id"])
                    # Write cleaned legacy profiles even when the new message has no stable facts.
                    changed = changed or bool(profile)
                    processed.append(marker)
                if changed:
                    await asyncio.to_thread(self._profile.upsert,
                        ids=["patient_profile:" + digest(user_id, patient_key)],
                        documents=[json.dumps(profile, ensure_ascii=False)],
                        metadatas=[{"domain": "healthcare", "user_id": user_id, "patient_key": patient_key,
                                    "updated_at": datetime.now().isoformat()}])
                # Mark only after successful persistence. Replays retain stable event IDs.
                for marker in processed:
                    await self._redis.setex(marker, 7 * 86400, "done")
