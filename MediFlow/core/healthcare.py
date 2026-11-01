"""Healthcare scenario policy, deterministic safety routing and demo knowledge.

These rules are a conservative routing aid, not a clinical triage classifier.
"""
import os
import re


def healthcare_enabled():
    """医疗是唯一运行场景；旧配置不能关闭医疗安全边界。"""
    scenario = os.getenv('MEDIFLOW_SCENARIO', 'healthcare').strip().lower()
    if scenario != 'healthcare':
        raise ValueError('MEDIFLOW_SCENARIO 仅支持 healthcare，请移除旧场景配置')
    return True


MEDICAL_POLICY = (
    '你是医疗健康咨询助手，仅提供健康信息参考，不替代医生诊断，不开处方。'
    '不得确定疾病、保证疗效、擅自建议停药换药或调整个人剂量；不评价其他医生的诊疗方案。'
    '用药禁忌、相互作用和用法必须核对具体药品通用名、规格、说明书及可靠资料；'
    '缺少依据时明确不能确认，建议联系医生或药师，不得凭模型记忆编造剂量。'
    '儿童、孕哺期、过敏史、肝肾疾病等需要专业核验。'
    '报告指标必须结合单位、参考范围、年龄及临床背景，单个异常值不能确诊。'
    '挂号仅提供流程指引，不声称查询到号源、预约成功、接通医生或已呼叫急救。'
    '不要索取身份证号、住址、手机号、支付凭证或未经脱敏的完整报告。'
    '发现正在发生的胸痛、呼吸困难、意识异常等危险信号，优先提示立即就医及中国大陆120；'
    '不要为问诊、检索、挂号或等待回复而延误急救。'
)

RED_FLAGS = (
    '胸痛', '胸口痛', '呼吸困难', '喘不过气', '呼吸不畅', '无法呼吸',
    '意识丧失', '意识不清', '意识模糊', '昏迷', '晕倒', '叫不醒',
    '口唇发紫', '突然口角歪斜', '突然一侧无力', '抽搐', '严重出血',
    'chest pain', 'difficulty breathing', 'cannot breathe', 'unconscious',
)


def _has_unnegated_mention(text, symptom):
    remaining = re.sub(
        r'(?:没有|没出现|未出现|无明显|并无|否认|不伴有|不伴|没有出现|无|不)(?:任何|明显)?\s*' + re.escape(symptom),
        '', text,
    )
    remaining = re.sub(r'\b(?:no|without|not having)\s+' + re.escape(symptom), '', remaining)
    return symptom in remaining


def emergency_signals(message):
    text = str(message or '').lower()
    # Pure educational / hypothetical questions do not assert a current emergency.
    clauses = re.split(r'[，,。！？!?；;\n]', text)
    text = '，'.join(clause for clause in clauses if not re.search(
        r'什么是|科普|如何识别|怎样识别|预防|假如|如果|假设|有哪些症状|哪些症状|what is|how to recognize', clause))
    return [flag for flag in RED_FLAGS if _has_unnegated_mention(text, flag)]


def medical_rule_intent(message):
    text = str(message or '').lower()
    if emergency_signals(text):
        return 'emergency'
    if re.search(r'相互作用|一起吃|一起服|同时服|联合用药|药物.*搭配|drug interaction', text):
        return 'medication_interaction'
    if re.search(r'不良反应|副作用|(?:吃药|服药|用药|服用|吃了).{0,20}(?:后|引起|导致).{0,12}(?:皮疹|恶心|头晕|呕吐|过敏|不舒服)|adverse reaction', text):
        return 'medication_adverse_reaction'
    if re.search(r'用法用量|剂量|吃几片|吃多少|一天几次|一次几|漏服|漏吃|dosage', text):
        return 'medication_dosage'
    if re.search(r'说明书|药物|药品|用药|吃药|服药|剂量|禁忌|相互作用|一起吃|布洛芬|阿莫西林|对乙酰氨基酚|头孢|medication|medicine', text):
        return 'medication_query'
    if re.search(r'报告|化验|血常规|检查结果|参考范围|指标|转氨酶|检验|白细胞|report', text):
        if re.search(r'偏高|偏低|升高|降低|超标|异常|不正常|白细胞.{0,8}[高低]|abnormal', text):
            return 'report_abnormal'
        return 'report_query'
    if re.search(r'哪个科|什么科|哪科|选择.{0,6}科室|科室选择|看.{0,4}科|department', text):
        return 'department_query'
    if re.search(r'挂号|预约|号源|就诊流程|就诊材料|医保|哪个科|什么科|科室|appointment', text):
        return 'appointment'
    if re.search(r'症状|体温|发烧|发热|咳嗽|头痛|头疼|腹痛|肚子疼|恶心|头晕|过敏|疼痛|不舒服|胸痛|呼吸困难|fever|symptom', text):
        return 'symptom_query'
    return None


def medical_entities(message):
    from core.medical_entities import extract
    return extract(message)


def high_fever(message):
    return any(float(re.search(r'\d+(?:\.\d+)?', value).group()) >= 39.5
               for value in re.findall(r'(?<![\d.])(?:3\d|4[0-3])(?:\.\d+)?\s*(?:℃|°C|度(?!酒))',
                                       str(message or ''), re.I))


def emergency_reply():
    return (
        '你描述的情况可能涉及急症，需要立即由急救人员或急诊医生评估。'
        '中国大陆请立即拨打120；在其他地区请联系当地急救电话。'
        '不要等待在线咨询、挂号或报告解读，也不要自行驾车。'
        '尽量请身边的人陪同，并向调度员说明位置、症状、发生时间和意识/呼吸情况，按调度员指导行动。'
        '本助手不能诊断，也未替你呼叫救护车或接通医生。'
    )


HEALTH_DOCS = [
    {'document_id': 'health-emergency-v1', 'title': '急症识别与就医提醒（健康教育）', 'content':
     '突发胸痛、呼吸困难、昏迷等危险表现应立即寻求急救；中国大陆急救电话120。不要等待在线问答或挂号。'
     '普通发热不能仅凭体温确诊。高热持续、病情加重或高风险人群应及时联系医生；伴危险信号时立即急救。'
     '来源：国家卫生健康委2024年11月21日新闻发布会，https://www.nhc.gov.cn/xcs/c100122/202411/81a60171b43d43ff98cc6110d65a4136.shtml；'
     '成人发热参考：https://www.nhs.uk/symptoms/fever-in-adults/ 。此资料不替代诊断。'},
    {'document_id': 'health-medication-v1', 'title': '用药咨询前的信息核对', 'content':
     '用药咨询先核对药品通用名、剂型、规格、已开具的用法、年龄、过敏史及全部合并用药。'
     '相同商品名或不完整药名不能用于确认相互作用。不得自行加量、停药或换药。'
     '儿童、孕哺期及肝肾疾病患者应由医生或药师核验。没有对应说明书时不能给出具体剂量或宣称联合用药安全。'
     '来源：英国公立医院患者用药资料 https://www.chelwest.nhs.uk/your-visit/patient-leaflets/taking-medicines 。'
     '这是咨询准备指南，不是某种药品的说明书或相互作用数据库。'},
    {'document_id': 'health-appointment-v1', 'title': '挂号与就诊材料准备（通用指引）', 'content':
     '通过医院官方公众号、官网、App或挂号窗口查看科室介绍、出诊安排和号源。信息不确定时咨询医院导诊台。'
     '准备就诊人所需证件或医保凭证、既往相关报告和用药清单；所需材料、费用和预约规则以目标医院官方说明为准。'
     '在本助手中无需上传身份证、医保卡或支付信息。本系统未接入医院挂号系统，不能查询实时号源或完成预约。'
     '急症优先急诊或急救，不能为等待普通门诊挂号延误处理。此文是产品流程示例，不代表任何医院的实际规则。'},
    {'document_id': 'health-report-v1', 'title': '检查报告解读的边界与输入格式', 'content':
     '可提供脱敏后的指标名称、数值、单位、检验机构参考范围及检查日期，并说明年龄和相关症状。'
     '不同实验室、检测方法和人群的参考范围可能不同，单个偏高或偏低指标不能独立诊断疾病。'
     '本助手只解释报告信息及建议向医生确认的问题；不能替代医生结合病史、查体及其他检查作出判断。'
     '不要上传姓名、身份证号、条码或完整未脱敏报告。此文是报告咨询流程示例，不提供指标诊断阈值。'},
]
