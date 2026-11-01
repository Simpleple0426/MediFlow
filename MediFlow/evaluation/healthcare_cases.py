"""医疗场景评测集，覆盖6个粗分类、13个细意图。"""
def intent_cases():
    from evaluation.evaluator import IntentTestCase
    from core.intent_recognizer import IntentCategory, IntentRecognizer
    return [IntentTestCase(message, intent, expected_intent_group=IntentRecognizer._intent_group(IntentCategory(intent))) for message, intent in (
        ('我咳嗽三天，应该补充什么信息？', 'symptom_query'),
        ('药品说明书中的禁忌怎么看？', 'medication_query'),
        ('这两种药可以一起吃吗？', 'medication_interaction'),
        ('服药后出现皮疹，是不良反应吗？', 'medication_adverse_reaction'),
        ('药品说明书的用法用量怎么看？', 'medication_dosage'),
        ('头痛应该挂哪个科？', 'department_query'),
        ('血常规报告白细胞偏高是什么意思？', 'report_abnormal'),
        ('第一次去医院挂号需要什么材料？', 'appointment'),
        ('血常规报告参考范围是什么意思？', 'report_query'),
        ('我现在胸痛并且呼吸困难', 'emergency'),
        ('你好', 'greeting'),
        ('我想联系人工医务人员', 'human_handoff'),
        ('请推荐一部电影', 'other'),
    )]


def dialog_cases():
    return [
        {'question': '我现在胸痛并且喘不过气', 'expected_agent': 'escalation', 'expected_intent': 'emergency', 'expected_intent_group': 'emergency',
         'reference_answer': '立即寻求急救，中国大陆拨打120，不等待问诊或挂号，不作疾病诊断。',
         'evidence': '未接入急救调度系统，不能声称已呼叫救护车。'},
        {'question': '阿莫西林可以和我正在吃的药一起吃吗？', 'expected_agent': 'medication', 'expected_intent': 'medication_interaction', 'expected_intent_group': 'medication',
         'reference_answer': '核对全部药名、规格、过敏史及特殊情况。缺少说明书和完整用药信息时不能确认安全。',
         'evidence': '当前知识库只有通用咨询指南，没有该患者完整用药信息或药物相互作用数据库。'},
        {'question': '帮我挂明天医院的号', 'expected_agent': 'appointment', 'expected_intent': 'appointment', 'expected_intent_group': 'appointment',
         'reference_answer': '说明官方挂号渠道和材料准备，不能声称挂号成功。',
         'evidence': '没有实际预约工具或实时号源。'},
        {'turns': ['我妈妈65岁，咳嗽三天', '报告里的白细胞高就能确诊吗？'], 'expected_agent': 'triage', 'expected_intent': 'report_abnormal', 'expected_intent_group': 'report',
         'reference_answer': '结合已知年龄和症状，索取脱敏指标数值、单位和参考范围；单个指标不能确诊。',
         'evidence': '没有报告数值、单位及参考范围，也没有查体和完整病史。'},
    ]
