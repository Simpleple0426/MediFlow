import { pollEvaluation } from './evaluationJobs.js'
const DEFAULT_BACKENDS = {
  python: {
    id: 'python',
    label: 'Python',
    baseUrl: runtimeConfig().pythonApiUrl || import.meta.env.VITE_PYTHON_API_URL || '/api/python',
    port: '8000'
  },
  java: {
    id: 'java',
    label: 'Java',
    baseUrl: runtimeConfig().javaApiUrl || import.meta.env.VITE_JAVA_API_URL || '/api/java',
    port: '8080'
  }
}

export const enabledBackends = String(runtimeConfig().enabledBackends || import.meta.env.VITE_ENABLED_BACKENDS || 'python,java')
  .split(',').map(value => value.trim()).filter(value => value in DEFAULT_BACKENDS)
if (!enabledBackends.length) enabledBackends.push('python')

export function createInitialSettings() {
  const saved = readSettings()
  return {
    pythonToken: '',
    backend: enabledBackends.includes(saved.backend) ? saved.backend :
      enabledBackends.includes(import.meta.env.VITE_DEFAULT_BACKEND) ? import.meta.env.VITE_DEFAULT_BACKEND : enabledBackends[0],
    userId: saved.userId || 'u1001',
    patientId: saved.patientId || '',
    conversationId: saved.conversationId || '',
    endpoints: {
      python: saved.endpoints?.python || DEFAULT_BACKENDS.python.baseUrl,
      java: saved.endpoints?.java || DEFAULT_BACKENDS.java.baseUrl
    }
  }
}

export function saveSettings(settings) {
  localStorage.setItem('mediflow.healthcare.settings', JSON.stringify({ ...settings, pythonToken: undefined }))
}

export function backendMeta(type, settings) {
  const meta = DEFAULT_BACKENDS[type] || DEFAULT_BACKENDS.java
  return {
    ...meta,
    baseUrl: normalizeBaseUrl(settings.endpoints[type] || meta.baseUrl)
  }
}

export async function requestHealth(type, settings) {
  return requestFor(type, settings, '/health')
}

export async function requestMonitor(type, settings) {
  return requestFor(type, settings, '/monitor')
}

export async function requestSkills(type, settings) {
  return requestFor(type, settings, '/skills')
}

export async function reloadSkills(type, settings) {
  return requestFor(type, settings, '/skills/reload', { method: 'POST' })
}

export async function requestKnowledgeStats(type, settings) {
  return requestFor(type, settings, '/knowledge/stats')
}

export async function runEvaluation(type, settings, body = null, options = {}) {
  const taskKey = `mediflow.eval:${type}:${backendMeta(type, settings).baseUrl}:${settings.userId}`
  const savedJob = type === 'python' ? sessionStorage.getItem(taskKey) : null
  const initial = savedJob ? {job_id: savedJob, status: 'queued', completed: 0, total: 0} : await requestFor(type, settings, '/eval/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  })
  if (!initial.job_id) return initial // Java 后端仍兼容同步报告。
  sessionStorage.setItem(taskKey, initial.job_id)
  try {
    const result = await pollEvaluation(initial, jobId => requestFor(type, settings,
      `/eval/jobs/${encodeURIComponent(jobId)}`), options)
    if (result) sessionStorage.removeItem(taskKey)
    return result
  } catch (error) {
    if (error.terminal || error.status === 404) sessionStorage.removeItem(taskKey)
    throw error
  }
}

export async function requestSearch(type, settings, query, topK = 5) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  return requestFor(type, settings, `/search?${params}`, { method: 'POST' })
}

export async function requestChat(type, settings, message) {
  const meta = backendMeta(type, settings)
  const payload = buildChatPayload(type, settings, message)
  const attemptKey = JSON.stringify([type, meta.baseUrl, settings.userId, settings.patientId || '', settings.conversationId, message])
  if (type === 'python') {
    if (!chatAttempts.has(attemptKey)) {
      if (chatAttempts.size > 100) chatAttempts.clear()
      chatAttempts.set(attemptKey, crypto.randomUUID())
    }
    payload.client_request_id = chatAttempts.get(attemptKey)
  }
  const raw = await requestFor(type, settings, '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  chatAttempts.delete(attemptKey)
  return normalizeChatResponse(type, raw)
}

export async function requestToolTrace(type, settings, requestId) {
  if (!requestId) return null
  const raw = await requestFor(type, settings, `/trace/tool/${encodeURIComponent(requestId)}`)
  return normalizeToolTraceResponse(raw)
}

export async function addKnowledge(type, settings, documents) {
  return requestFor(type, settings, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(type, settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestFor(type, settings, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

function buildChatPayload(type, settings, message) {
  return {
    message,
    user_id: settings.userId || 'anonymous',
    patient_id: type === 'python' ? settings.patientId || undefined : undefined,
    conv_id: settings.conversationId || undefined
  }
}

function normalizeChatResponse(type, raw) {
  return {
    backend: type,
    conversationId: raw.conversation_id || raw.conversationId || raw.conv_id || '',
    requestId: raw.request_id || raw.requestId || '',
    response: raw.response || '',
    status: raw.status || 'success',
    retryable: Boolean(raw.retryable),
    intent: raw.intent || 'other',
    intentGroup: raw.intent_group || raw.intentGroup || 'general',
    agentType: raw.agent_type || raw.agentType || '',
    agentTypes: raw.agent_types || raw.agentTypes || [],
    primaryAgent: raw.primary_agent || raw.primaryAgent || '',
    supportingAgents: raw.supporting_agents || raw.supportingAgents || [],
    routingReason: raw.routing_reason || raw.routingReason || '',
    routingConfidence: Number(raw.routing_confidence ?? raw.routingConfidence ?? 0),
    entities: raw.entities || {},
    intentConfidence: Number(raw.intent_confidence ?? raw.intentConfidence ?? 0),
    intentSourceScores: raw.intent_source_scores || raw.intentSourceScores || {},
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    verified: raw.verified,
    grounded: raw.grounded,
    raw
  }
}

function normalizeToolTraceResponse(raw) {
  const trace = raw?.trace || {}
  return {
    requestId: raw?.request_id || raw?.requestId || '',
    found: Boolean(raw?.found),
    trace: {
      ...trace,
      toolsUsed: trace.tools_used || trace.toolsUsed || [],
      toolCalls: trace.tool_calls || trace.toolCalls || []
    },
    raw
  }
}

const chatAttempts = new Map()
async function requestFor(type, settings, path, options = {}) {
  const headers = new Headers(options.headers || {})
  if (type === 'python' && settings.pythonToken) headers.set('Authorization', `Bearer ${settings.pythonToken}`)
  return requestJson(backendMeta(type, settings).baseUrl, path, { ...options, headers,
    signal: options.signal || AbortSignal.timeout(path === '/chat' ? 55000 : 20000) })
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    const patientConflict = path === '/chat' && response.status === 422 && data?.detail?.code === 'patient_scope_conflict'
    const error = new Error(patientConflict
      ? '患者编号与咨询对象不一致。请在连接配置中更换或清空患者记忆编号后重试。'
      : path === '/chat'
      ? (response.status === 401 || response.status === 403 ? '访问验证未通过，请检查连接配置。'
        : '健康咨询服务暂时不可用，请联系医生或药师；如有急症请立即就医。')
      : `${response.status} ${response.statusText}: ${detail}`)
    error.status = response.status
    error.detail = data
    throw error
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem('mediflow.healthcare.settings') || '{}')
  } catch {
    return {}
  }
}

function runtimeConfig() {
  if (typeof window === 'undefined') return {}
  return window.__MEDIFLOW_CONFIG__ || {}
}
