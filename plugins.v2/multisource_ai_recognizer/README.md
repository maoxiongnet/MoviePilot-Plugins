# MultisourceAIRecognizer（多源AI识别与评分）

- **作用**：增强 MoviePilot 的名称识别（仅当主程序无法识别时触发）。
- **方法**：LLM（DeepSeek/OpenAI 兼容 JSON 模式）解析标题，联合 Douban/Trakt/Bangumi/TMDB 进行多源互证，产出**可超过 100** 的积分制评分。
- **SLA**：≤15 秒。低分将“空结果返回 + 进入人工队列”，避免阻塞主流程。

## 安装与配置
1. 将本插件目录放到第三方插件仓库 `plugins.v2/multisource_ai_recognizer/` 下。
2. 在 MoviePilot 前端 **插件市场** 添加该仓库地址并安装。
3. 在插件配置页填写：
   - `LLM Base/Model/Key`（如：`https://api.gptapi.us/v1` + `deepseek-v3`）
   - （可选）`Trakt API Key`、`TMDB API Key`
   - `MP API Base`（一般 `http://127.0.0.1:3001`）与 `MP API Token`
   - 阈值：`自动通过 ≥120`，`人工队列 80–119`，`<80 强制人工`

> **为什么强制 JSON 模式**：你的兼容网关仅 `response_format=json_object` 可用；tools/functions 不可用。此插件不依赖 tools。

## 打分规则（默认，可在配置页覆盖权重）
- **ID 命中**：tmdb +60，douban +50，bangumi +55，trakt +40
- **标题相似度**：max 相似度 ×30
- **年份匹配**：同年 +15；±1 年 +9；冲突 −30
- **季/集匹配**：+25
- **多源一致性**：两两一致每对 +12，上限 +36
- **类型/地区加成**：动漫命中 Bangumi +10；电影命中 IMDb/TMDB +10
- **AI（LLM）贡献**：
  - 结构化输出有效：+15
  - 字段完备度：0–15（按 name/year/season/episode/resolution/version/part 填充比例）
- **惩罚**：仅自然语言、无结构化字段 −20

**推荐阈值**：
- `自动通过 ≥ 120`
- `人工确认 80 – 119`
- `< 80 强制人工`

## 人工干预（可选目录）
- 插件页面提供“候选队列”，点“确认并整理”后：
  1. 通过 `/api/v1/storage/list` 浏览目录、`/api/v1/storage/mkdir` 新建目录；
  2. 提交到 `/api/v1/transfer/manual`，携带 `target_storage/target_path` 与基本 `mediainfo` 完成入库。

## 常见问题
- **为什么有时返回空结果？**
  - 为满足 15 秒窗口与“以先到为准”的规则，低分/不确定时会立刻空结果返回，同时把候选放到人工队列，避免误识别与阻塞。
- **AI 的分数是多少？**
  - 结构化 +15；字段完备 0–15；若与任一外部源一致，还能在“一致性/相似度/年份”等项拿到额外加分。
- **分数可以超过 100 吗？**
  - 可以。本插件采用积分制，>100 很常见；阈值用绝对分数判断。
