# MediFlow 医疗健康咨询助手

MediFlow 是基于 Python、FastAPI 与 Vue 的医疗健康咨询项目，提供症状信息收集、药品资料核验、就诊流程指引、报告指标预解读和急症提示。系统通过意图识别、多 Agent 协作、知识检索与患者范围记忆组织咨询过程。

本项目仅提供信息参考，不提供疾病确诊、处方、真实挂号或急救呼叫。出现危险信号时应及时联系当地急救服务。

## 主要能力

- 三路意图识别：模型、向量匹配和规则融合，覆盖 6 个粗分类、13 个细意图。
- 五类 Agent：分诊、用药、就诊、通用咨询与升级处理。
- 医疗规则以 `MediFlow/skills/healthcare/*.json` 配置，支持运行时加载。
- Redis 保存工作记忆，ChromaDB 支持资料检索、会话归档与长期画像。
- 按用户和患者编号隔离记忆；同一账号与患者编号可跨会话读取既往信息。
- 前端展示咨询、请求状态、工具调用轨迹、知识管理与场景评测；Prometheus 提供运行指标。

## 项目结构

```text
MediFlow/
  agents/       Agent 编排与医疗工具
  api/          FastAPI 接口
  core/         意图、模型传输、患者信息与输出约束
  memory/       会话及长期记忆
  tooling/      知识库与 MCP 集成
  evaluation/   应用内场景评测功能
  monitor/      运行监控
  config/       网关、监控与 MCP 配置
  skills/       医疗规则 JSON
MediFlowFrontend/
  src/          Vue 页面及连接逻辑
  docker/       前端运行配置
start.ps1       Windows 本机启动入口
```

## Windows 快速启动

安装 Docker Desktop，启用 Linux containers。在仓库根目录执行：

```powershell
Copy-Item .\MediFlow\.env.example .\MediFlow\.env
```

编辑 `MediFlow/.env`，填写模型配置。默认支持硅基流动：

```dotenv
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=填写你的密钥
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V4-Flash
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

也可设置 `LLM_PROVIDER=openai`，并填写 `OPENAI_API_KEY`、`OPENAI_MODEL` 与 `OPENAI_BASE_URL`，连接兼容 Chat Completions 的服务。模型密钥只放在后端 `.env`，不要写入前端或提交到仓库。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
.\start.ps1 status
.\start.ps1 logs
.\start.ps1 stop
```

首次构建会下载依赖和向量模型，需要可用网络。没有模型密钥时可启动页面与急症提示，普通 AI 咨询需要有效密钥和额度。

| 入口 | 地址 |
| --- | --- |
| 前端页面 | http://localhost:5175 |
| API 文档 | http://localhost:8002/docs |
| 健康检查 | http://localhost:8002/health |
| API 网关 | http://localhost:8088/docs |
| Prometheus | http://localhost:9090 |

停止服务会保留本机数据。修改代码后重新运行启动脚本完成构建；仅修改环境配置时可用 `./start.ps1 -NoBuild`。

## 前端开发

先启动后端，再执行：

```powershell
cd MediFlowFrontend
npm ci
npm run dev
```

开发页面为 `http://localhost:5173`，默认代理后端 `http://localhost:8002`。运行 `npm run build` 生成前端产物。

## 咨询接口

通过 API 文档调用 `POST /chat`，建议使用虚构示例信息：

```json
{
  "message": "我妈妈65岁，咳嗽三天，应该补充什么信息？",
  "user_id": "health_demo",
  "patient_id": "demo_patient",
  "client_request_id": "health-demo-001"
}
```

切换患者时使用不同患者编号，避免混淆资料。医疗画像记录用户陈述，不将模型推测自动视为已确诊事实。

## 运行配置

- `MEDIFLOW_SCENARIO=healthcare`：仅支持医疗场景。
- `MEDIFLOW_SKILLS_DIR`：医疗规则路径；本机部署使用容器内 `/app/skills/healthcare`。
- `INTENT_EMBEDDING_PROVIDER=auto`：有配置时使用在线意图向量，无配置时使用本地匹配；不替换知识库检索模型。
- `MCP_EXTERNAL_ENABLED=false`：默认关闭外部 MCP 服务。
- `MEDIFLOW_ENV=development`：本机默认值。对外部署应设为 `production`，配置 `MEDIFLOW_API_TOKENS` 和独立 Redis 密码，并核对访问控制与网络入口。

完整环境变量模板见 `MediFlow/.env.example`。本机密钥、运行数据、缓存、测试文件及额外文档不随本仓库发布。
