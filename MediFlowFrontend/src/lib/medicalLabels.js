export const INTENT_LABELS = {
  symptom_query: '症状咨询', medication_query: '药品信息', medication_interaction: '药物相互作用',
  medication_adverse_reaction: '疑似不良反应', medication_dosage: '用法用量',
  report_query: '报告指标解释', report_abnormal: '报告异常咨询',
  appointment: '挂号流程', department_query: '科室选择', emergency: '危险信号',
  greeting: '问候', human_handoff: '人工协助', other: '其他需求',
}
export const GROUP_LABELS = {
  symptom: '症状', medication: '用药', report: '报告', appointment: '就诊', emergency: '急症', general: '通用',
}
export const AGENT_LABELS = {
  triage: '预检分诊', medication: '用药咨询', appointment: '挂号指引', escalation: '就医与人工提醒', general: '健康接待',
}
export function formatIntent(intent, group) {
  return [GROUP_LABELS[group] || group, INTENT_LABELS[intent] || intent].filter(Boolean).join(' · ')
}
