"""Stable profile schema, lazy legacy cleanup and evidence-preserving updates."""
from copy import deepcopy

from core.medical_entities import preferences, rule_allergies
from core.patient_identity import digest

NOTICE = "历史用户陈述及表达偏好，未经医学核验；年龄以记录日期为准，使用前确认时效，不自动视为当前状态。"


def allergy_view(events):
    latest = {}
    for event in sorted(events, key=lambda x: x.get("reported_at", "")):
        name = event["substance"]
        old = latest.get(name)
        conflict = bool(old and old["status"] != event["status"])
        latest[name] = {**event, "needs_confirmation": bool(event.get("legacy") or
            event["status"] != "withdrawn" and (conflict or (old or {}).get("needs_confirmation")))}
    generic = latest.get("")
    for name, item in latest.items():
        if name and generic and item["status"] == "reported" and generic["status"] == "denied":
            item["needs_confirmation"] = generic["needs_confirmation"] = True
    return list(latest.values())


def sanitize_profile(value, label):
    """Whitelist stable data; never expose legacy symptoms/drugs/report fields."""
    value = value if isinstance(value, dict) else {}
    if not value:
        return {}
    reported = value.get("reported_entities", {})
    fields = {key: deepcopy(reported[key]) for key in ("patient", "age")
              if isinstance(reported.get(key), dict)}
    events = deepcopy(value.get("allergy_history", [])) if value.get("schema_version") == 2 else []
    # Existing literal allergy reports survive as unverified legacy evidence.
    if value.get("schema_version") != 2:
        old = reported.get("allergy", {})
        for text in old.get("values", []):
            for fact in rule_allergies(text, label):
                events.append({**fact, "event_id": "legacy:" + digest(text), "legacy": True,
                    "reported_at": old.get("reported_at", ""), "conv_id": old.get("conv_id", "")})
    style = []
    for item in value.get("preferences", []):
        if isinstance(item, dict):
            for text in preferences(item.get("text", "")):
                style.append({"text": text, **{k: item[k] for k in ("reported_at", "conv_id", "request_id") if k in item}})
    return {"schema_version": 2, "patient_label": label, "reported_entities": fields,
            "allergy_history": events, "allergies": allergy_view(events),
            "preferences": style, "notice": NOTICE}


def apply_patch(profile, patch, scope, stamp, conv_id, request_id):
    profile = sanitize_profile(profile, scope["label"]) or {
        "schema_version": 2, "patient_label": scope["label"], "reported_entities": {},
        "allergy_history": [], "allergies": [], "preferences": [], "notice": NOTICE}
    source = {"reported_at": stamp, "conv_id": conv_id, "request_id": request_id}
    for field in ("patient", "age"):
        values = patch.get(field, [])
        old = profile["reported_entities"].get(field, {})
        if values and stamp >= old.get("reported_at", ""):
            profile["reported_entities"][field] = {"values": values, **source}
    for fact in patch.get("allergies", []):
        event = {**fact, **source, "event_id": digest(conv_id, request_id, fact)}
        if not any(old["event_id"] == event["event_id"] for old in profile["allergy_history"]):
            profile["allergy_history"].append(event)
    profile["allergies"] = allergy_view(profile["allergy_history"])
    for text in patch.get("preference", []):
        old = next((p for p in profile["preferences"] if p["text"] == text), None)
        if old and old.get("reported_at", "") > stamp:
            continue
        profile["preferences"] = [p for p in profile["preferences"] if p["text"] != text]
        profile["preferences"].append({"text": text, **source})
    profile["preferences"] = sorted(profile["preferences"], key=lambda x: x.get("reported_at", ""))[-10:]
    return profile
