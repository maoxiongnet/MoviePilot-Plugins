# 多源AI识别与评分（Multisource AI Recognizer）

LLM（**严格 JSON 模式**）+ Douban/Trakt/Bangumi/TMDB 多源互证  
积分制评分（**允许 >100**），仅在“**不确定/模糊**”时触发 AI（可切换为总是/手动）。  
支持：**自动下载**（≥阈值可选）、**人工队列**（自选目录落地）、**批量问AI（全部/所选）**、**🧪一键自检**。

> 推荐阈值：自动 ≥ **120**；人工 **80–119**；强制人工 < **80**。  
> AI 贡献：结构化 **+15**；字段完备 **0–15**。

---

## 版本/依赖

- MoviePilot：建议 ≥ **v2.5.7**（按钮回调/页面 DSL 更稳；低版本也能用主要功能）
- Python 依赖：`requests>=2.31.0`（见 `requirements.txt`）

---

## 安装

1. 将本插件放到 `plugins.v2/multisource_ai_recognizer/`
2. `requirements.txt` 中 `requests` 会被自动安装（或手动安装）
3. 在 WebUI 启用插件

---

## 配置

在插件“设置”页：

- **LLM**
  - `llm_base`：如 `https://api.gptapi.us/v1`
  - `llm_model`：如 `deepseek-v3`
  - `llm_key`：你的密钥  
  > 必须支持 OpenAI 兼容的 `/chat/completions` 且 `response_format={"type":"json_object"}`

- **行为**
  - `ask_mode`：`smart`（默认，仅不确定时问 AI） / `always` / `manual`
  - `auto_download`：勾选后，当总分 ≥ `threshold_auto` 时自动下载
  - `threshold_auto`：默认 `120`
  - `threshold_manual`：默认 `80`

- **主程序直连（可选）**
  - `mp_api_base`、`mp_api_token`：留空时，**页面用相对路径** `api/v1/...` 调用主程序；填写后，插件后端也可直连（自动下载/整理会直接提交）

> **费用控制**：默认 `smart`，只有不确定才问 AI；页面还提供“**AI识别（全部/所选）**”按钮，按需触发。

---

## 使用

### 1）数据页（队列 + 批量AI + 目录确认）
- 顶部：
  - **🤖 AI识别（全部）**：对队列中未有 AI 结构化的条目全部请求一次
  - **🤖 AI识别（所选）**：只对勾选项请求，省钱
  - **AI 触发模式**：`smart/always/manual` 切换
  - **目标存储/目录**：输入 `target_storage`、`target_path`（如 `local` + `/media/Movies`）
- 列表：
  - 每行“**确认并整理**” → 生成 `transfer/manual` 的 payload 并提交（直连已配则后端直提，没配则返回给前端相对路由提交）
- 可选：在设置中打开 **自动下载**（≥自动阈值即触发）

### 2）🧪 一键自检
- 点击 **“🧪 自检”** 按钮，会跑 4 项：
  1. **LLM 兼容性**：`/chat/completions + json_object` 能否解析
  2. **主程序 API 直连**（如已配置）
  3. **打分器**：用示例数据跑一遍
  4. **队列→确认流程**：模拟生成 `transfer/manual` 的 body  
- 结果与**详细日志**会在页面展示，便于排查

---

## 打分项（默认权重）

- **ID命中**：tmdb +60 / douban +50 / bangumi +55 / trakt +40  
- **标题相似度**：0~30（多语别名取最高）  
- **年份**：精确 +15；±1 年 +9；冲突 −30  
- **季/集**：最多 +25  
- **一致性**：两两一致每对 +12，上限 +36  
- **领域加成**：动漫+Bangumi +10；电影（IMDb+TMDB）+10  
- **AI 贡献**：结构化 +15；字段完备 0~15  
- **惩罚**：非结构化 −20；ID 冲突 −50

> 权重可在配置里按字典覆盖。分数可 **>100**（例如多源一致+领域加成同时命中）。

---

## 接口（供二次开发/调试）

页面通过**相对路由**调用本插件 API：
- `plugin/multisource_ai_recognizer/queue`（GET）：获取人工队列
- `plugin/multisource_ai_recognizer/config`（POST）：更新 `ask_mode/阈值/自动下载`
- `plugin/multisource_ai_recognizer/ai_batch`（POST）：批量 AI 识别（`{"scope":"all"}` 或 `{"scope":"selected","ids":[...]}`）
- `plugin/multisource_ai_recognizer/confirm`（POST）：确认并整理（`{"id","target_storage","target_path"}`）
- `plugin/multisource_ai_recognizer/selftest`（POST）：**一键自检**

对主程序的调用在页面使用**相对路径**：
- `api/v1/transfer/manual`
- `api/v1/download/`
- `api/v1/storage/list`
- `api/v1/storage/mkdir`

> 如配置了 `mp_api_base/mp_api_token`，插件后端也会直连上述接口。

---

## 常见问题

- **为什么不总是问 AI？**  
  省钱也更稳。`smart` 只在 MP 自身识别/搜索不唯一或缺关键信息时触发 AI。

- **第三方 OpenAI 网关不兼容？**  
  需要支持 `response_format={"type":"json_object"}`。若不支持，**🧪 自检**会直接提示失败。

- **自动下载未生效？**  
  确认：1）`auto_download` 已开启；2）总分确实 ≥ `threshold_auto`；3）如走后端直连，`mp_api_base/mp_api_token` 需正确；否则会把下载 payload 放到队列让前端提交。

---

## 变更日志

- **1.3.0**
  - 新增 **🧪 一键自检**（LLM/主程序 API/评分/流程）+ 详细日志
  - 数据页加入 **批量 AI（全部/所选）** 按钮
  - 默认 `ask_mode=smart`，只在不确定时调用 LLM
  - 稳健化页面交互（不依赖弹窗也可直接确认并整理）
- **1.2.x**
  - 初版：积分制、人工队列、仪表板、工作流、消息命令
