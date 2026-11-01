"""按业务职责划分的只读工具，不连接医院或药品业务系统。"""
from datetime import date, datetime
import math
import unicodedata

from agents.tools import make_tool


TEXT = {"type": "string", "minLength": 1, "maxLength": 500, "pattern": r"\S"}
TEXTS = {"type": "array", "items": TEXT, "maxItems": 30, "uniqueItems": True}


def _object(properties, required):
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def _items(schema):
    return {"type": "array", "items": schema, "minItems": 1, "maxItems": 30}


def _key(text):
    # 仅规范字符形式和空白，绝不将商品名、别名或相似药名当成相同成分。
    return unicodedata.normalize("NFKC", text).strip().casefold()


def build_domain_tools():
    def route(req, args):
        from core.healthcare import emergency_signals, medical_rule_intent
        from agents.tools import consultation_plan
        message = args.get("message", req.message)
        intent = medical_rule_intent(message) or "other"
        role = ("escalation" if intent == "emergency" else
                "medication" if intent.startswith("medication") else
                "appointment" if intent in ("appointment", "department_query") else
                "triage" if intent in ("symptom_query", "report_query", "report_abnormal") else "general")
        return {"success": True, "suggested_agent": role, "intent_hint": intent,
                "matched_signals": emergency_signals(message), "routing_executed": False,
                "note": "使用现有关键词规则辅助接待，不改变实际编排路由；未命中不代表没有风险。",
                **consultation_plan(intent)}

    def completeness(req, args):
        from agents.tools import consultation_plan
        intent = args.get("intent") or (req.intent.value if req.intent else "other")
        plan = consultation_plan(intent)
        supplied = args["confirmed_fields"]
        missing = [field for field in plan["fields_to_confirm"] if field not in supplied]
        return {"success": True, **plan, "missing_fields": missing,
                "next_question_fields": missing[:2],
                "unmatched_fields": [field for field in supplied if field not in plan["fields_to_confirm"]],
                "note": "字段名须与咨询清单完全一致；仅检查字段覆盖，不验证内容真实性或医学充分性。"}

    def timeline(req, args):
        dated, undated = [], []
        for index, event in enumerate(args["events"]):
            if "occurred_at" not in event:
                undated.append({**event, "original_index": index})
                continue
            try:
                when = datetime.strptime(event["occurred_at"], "%Y-%m-%dT%H:%M")
            except ValueError:
                return {"success": False, "error": "时间须为有效的本地时间 YYYY-MM-DDTHH:MM；不确定时省略 occurred_at"}
            dated.append((when, index, event))
        ordered = [{**event, "original_index": index} for _, index, event in sorted(dated)]
        return {"success": True, "dated_events": ordered, "undated_events": undated,
                "note": "仅排序同一患者、同一时区已确认的事件；相对时间原文保留在 description 中，不推算未知日期或判断病因。"}

    def history(req, args):
        records = args["records"]
        try:
            dated = [(date.fromisoformat(row["date"]), index, row) for index, row in enumerate(records)]
        except ValueError:
            return {"success": False, "error": "检查日期必须是有效的 YYYY-MM-DD"}
        if any(not math.isfinite(row["value"]) for row in records):
            return {"success": False, "error": "数值必须为有限数字"}
        # 缺少检测机构或方法时不擅自认定可比。
        comparable = all(row.get("laboratory") and row.get("method") for row in records)
        comparable = comparable and len({tuple(row.get(k, "").strip() for k in
            ("name", "unit", "laboratory", "method")) for row in records}) == 1
        ordered = [row for _, _, row in sorted(dated)]
        comparisons = []
        if comparable:
            for previous, current in zip(ordered, ordered[1:]):
                delta = current["value"] - previous["value"]
                if not math.isfinite(delta):
                    return {"success": False, "error": "数值差超出可处理范围"}
                comparisons.append({"from_date": previous["date"], "to_date": current["date"],
                                    "delta": delta, "direction": "up" if delta > 0 else "down" if delta < 0 else "unchanged"})
        return {"success": True, "ordered_records": ordered, "comparable": bool(comparable),
                "comparisons": comparisons,
                "note": "仅计算同指标、同单位、同机构和同方法的相邻差值；不推断好转、恶化或临床意义。"}

    def ingredients(req, args):
        groups, unknown = {}, []
        for index, medicine in enumerate(args["medications"]):
            if not medicine.get("ingredients"):
                unknown.append(index)
            seen = set()
            for ingredient in medicine.get("ingredients", []):
                key = _key(ingredient)
                if key not in seen:
                    groups.setdefault(key, []).append({"index": index, "name": medicine["name"],
                                                       "ingredient_as_provided": ingredient})
                    seen.add(key)
        return {"success": True, "duplicate_ingredients": [
                    {"ingredient": key, "medications": entries} for key, entries in groups.items() if len(entries) > 1],
                "unknown_ingredient_indices": unknown, "interaction_checked": False,
                "note": "只对用户提供的说明书成分做精确文本查重；未发现重复不代表安全，不识别别名、剂量或相互作用，须由药师核验。"}

    def requirements(req, args):
        provided = {_key(item) for item in args["ready_materials"]}
        missing = [item for item in args["required_materials"] if _key(item) not in provided]
        return {"success": True, "requirements_source": args["requirements_source"],
                "missing_materials": missing, "ready_for_listed_requirements": not missing,
                "appointment_created": False, "official_requirements_verified": False,
                "note": "仅按提供的医院材料清单做名称匹配，不识别证件号码，不联网确认政策、资格或预约状态。"}

    return {spec.name: spec for spec in (
        make_tool("identify_consultation_route", "接待私有：用现有医疗关键词规则识别咨询方向和危险信号，返回建议角色；不执行路由。",
                  {"message": TEXT}, route, allowed_agents=("general",)),
        make_tool("check_consultation_completeness", "接待私有：将已确认字段名与当前咨询意图的字段清单比对，找出下一步需要澄清的项目。先用 prepare_health_consultation 获得标准字段名。",
                  {"intent": {"type": "string", "enum": ["symptom_query", "medication_query", "medication_interaction",
                    "medication_adverse_reaction", "medication_dosage", "report_query", "report_abnormal", "appointment",
                    "department_query", "greeting", "human_handoff", "emergency", "other"]}, "confirmed_fields": TEXTS},
                  completeness, ["confirmed_fields"], allowed_agents=("general",)),
        make_tool("build_symptom_timeline", "分诊私有：按同一时区已确认的绝对时间排列症状发生、变化和处置事件，未知时间单列；不推断病因。",
                  {"events": _items(_object({"description": TEXT, "occurred_at": {
                      "type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"}}, ["description"]))},
                  timeline, ["events"], allowed_agents=("triage",)),
        make_tool("compare_report_history", "分诊私有：排列同一患者历次报告；仅当指标、单位、机构及方法均一致时计算差值，不判断病情变化。",
                  {"records": {**_items(_object({"name": TEXT, "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                      "value": {"type": "number"}, "unit": TEXT, "laboratory": TEXT, "method": TEXT},
                      ["name", "date", "value", "unit"])), "minItems": 2}},
                  history, ["records"], allowed_agents=("triage",)),
        make_tool("find_duplicate_ingredients", "用药私有：对已提供的说明书成分做精确文本查重，返回涉及药品与缺失成分记录；不从药名猜成分，不核验合用安全。",
                  {"medications": _items(_object({"name": TEXT, "ingredients": TEXTS}, ["name"]))},
                  ingredients, ["medications"], allowed_agents=("medication",)),
        make_tool("check_visit_materials", "挂号私有：按用户提供的医院官方要求清单核对已备材料，列出缺项；不联网验证政策或预约。",
                  {"requirements_source": TEXT, "required_materials": {**TEXTS, "minItems": 1}, "ready_materials": TEXTS},
                  requirements, ["requirements_source", "required_materials", "ready_materials"], allowed_agents=("appointment",)),
    )}


def build_domain_search_tools(search):
    """复用现有 RAG 的改写、重排和错误处理，不制造额外知识来源。"""
    async def drug(req, args):
        topics = {"indications": "适应症", "contraindications": "禁忌", "interactions": "药物相互作用",
                  "adverse_reactions": "不良反应", "instructions": "说明书用法"}
        query = f"药品说明书 通用名：{args['drug_name']} 核对主题：{topics[args['topic']]}"
        if args.get("strength"):
            query += f" 规格：{args['strength']}"
        result = await search(req, {"query": query, "top_k": 5})
        return {**result, "drug_specific_evidence_verified": False,
                "note": "仅检索当前知识库，可能只有通用资料；须核对原文是否匹配具体药品及规格，不足时不能据此给剂量或合用结论。"}

    async def department(req, args):
        query = f"科室介绍 就诊范围 医院：{args['hospital']} 咨询问题：{args['concern']}"
        result = await search(req, {"query": query, "top_k": 5})
        return {**result, "hospital_specific_evidence_verified": False, "availability_checked": False,
                "note": "仅检索当前知识库，不访问医院实时系统；须核对原文是否确属目标医院，没有对应资料时联系官方导诊。"}

    return {spec.name: spec for spec in (
        make_tool("search_drug_information", "用药私有：按通用名、规格和核对主题专项检索现有知识库药品资料；结果未自动完成药品匹配或安全核验。",
                  {"drug_name": TEXT, "strength": TEXT, "topic": {"type": "string", "enum": [
                      "indications", "contraindications", "interactions", "adverse_reactions", "instructions"]}},
                  drug, ["drug_name", "topic"], allowed_agents=("medication",)),
        make_tool("search_department_guidance", "挂号私有：按目标医院与咨询问题专项检索现有知识库科室介绍；不编造目标医院政策、不查号源。",
                  {"hospital": TEXT, "concern": TEXT}, department, ["hospital", "concern"], allowed_agents=("appointment",)),
    )}
