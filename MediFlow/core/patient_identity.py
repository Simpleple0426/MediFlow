"""Patient scoping is independent of profile extraction and medical intent."""
import hashlib
import json
import re

RESET_PATIENT = re.compile(r"换个患者|另一位患者|换个人|重新咨询|换个问题")


def patient_labels(message):
    text = str(message or "")[:12000]
    labels = [name for name, aliases in (
        ("妈妈", ("妈妈", "母亲")), ("爸爸", ("爸爸", "父亲")),
        ("儿子", ("儿子",)), ("女儿", ("女儿",)),
        ("孩子", ("孩子", "宝宝", "婴儿")), ("本人", ("我本人", "我自己")),
    ) if any(alias in text for alias in aliases)]
    if re.search(r"(?:^|[，。；\n])\s*(?:现在(?:改)?问)?我(?=\d|[一二三四五六七八九十两]+岁|对|没有|无|的(?:报告|体温)|最近|正在|咳嗽|头痛)", text):
        labels.append("本人")
    if "儿子" in labels or "女儿" in labels:
        labels = [label for label in labels if label != "孩子"]
    return list(dict.fromkeys(labels))


class PatientScopeConflict(ValueError):
    """A patient ID cannot silently change to a different known person."""


def patient_family(label):
    if label in {"儿子", "女儿", "孩子"}:
        return "子女"
    return label if label in {"本人", "妈妈", "爸爸"} else None


def check_patient_label(old_label, new_label):
    old, new = patient_family(old_label), patient_family(new_label)
    if old and new and old != new:
        raise PatientScopeConflict(f"当前患者记忆编号已用于{old_label}；咨询{new_label}请更换编号，或清空编号后重新咨询。")


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=True).encode()).hexdigest()


def memory_scope(conv_id, message="", state=None, patient_id=None):
    """Relative labels are demo identities; use explicit IDs to distinguish siblings."""
    state = state if isinstance(state, dict) else {}
    previous = state.get("memory_scope", {})
    previous = previous if isinstance(previous, dict) else {}
    patients = patient_labels(message)
    reset = bool(RESET_PATIENT.search(message))
    if len(patients) > 1:
        return {"key": "session:" + digest(conv_id, "ambiguous"),
                "label": "多位患者待确认", "source": "session"}
    if patient_id:
        same_id = previous.get("key") == "patient:" + digest(patient_id)
        if same_id and patients:
            check_patient_label(previous.get("label"), patients[0])
        return {"key": "patient:" + digest(patient_id),
                "label": patients[0] if patients else previous.get("label", "指定患者") if same_id else "指定患者",
                "source": "explicit"}
    if previous.get("source") == "explicit" and not reset and (
            not patients or patients[0] == previous.get("label")):
        return dict(previous)
    if patients:
        return {"key": "relative:" + patients[0], "label": patients[0], "source": "relative"}
    if previous.get("key") and not reset:
        return dict(previous)
    old_patients = state.get("entities", {}).get("patient", [])
    if len(old_patients) == 1 and not reset:
        return {"key": "relative:" + old_patients[0], "label": old_patients[0], "source": "relative"}
    epoch = int(previous.get("epoch", 0)) + int(reset)
    return {"key": "session:" + digest(conv_id, epoch), "label": "未确认咨询对象",
            "source": "session", "epoch": epoch}
