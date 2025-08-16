# -*- coding: utf-8 -*-
"""
MultisourceAIRecognizer（多源AI识别与评分）
=================================================
- 挂载点：V2 链式事件 NameRecognize（仅在主程序无法识别时触发）。
- 目标：用 LLM + 多站点（Douban/Trakt/Bangumi/TMDB）交叉核验，给出稳定的结构化结果与“可超过 100 的积分制评分”。
- SLA：≤ 15s；低分时立刻空结果返回，同时把候选放入“人工确认队列”。

必要外链/规范：
- 插件结构与事件：见官方插件库 README（事件清单、NameRecognize/Result 与 15s 约束）。
- 插件页面/仪表板/配置：支持 JSON 配置与联邦远程组件（此处采用 JSON 配置页）。

使用建议：
- LLM：默认 DeepSeek 网关 `https://api.gptapi.us/v1` + `response_format=json_object`。
- 多源：
  - Trakt：需在本插件配置页填入 `trakt_api_key`（public key），可选 `trakt_token`（如需用户态接口）。
  - Bangumi：可无需 token 的搜索 API，若有 `bgm_token` 更佳。
  - Douban：优先走 MoviePilot 自带的 `/api/v1/media/search` 结果（由系统集成负责对接 Douban/TMDB 等），也可配置 `douban_cookie` 直连（非必须）。
  - TMDB：如配置了 `tmdb_api_key`，将用于补强年份/标题别名。

打分（默认阈值，可配置）：
- 自动通过阈值：≥ 120
- 进入人工队列：80–119
- 强制人工：< 80
- **AI（LLM）自身贡献**：结构化有效 +15；字段完备度 0~15；与任一外部源一致 +20（算在多源一致性中）

人工干预：
- 插件提供“候选队列”页面（`get_page()`），可逐条“确认并整理/下载”。
- 目录选择：通过 `/api/v1/storage/list` 浏览、`/api/v1/storage/mkdir` 新建，并在 `POST /api/v1/transfer/manual` 传 `target_storage/target_path` 完成“自选目录”的入库/转移。

"""
from __future__ import annotations
import os
import re
import json
import time
import math
import uuid
import queue
import shutil
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

import requests

# ===== MoviePilot 插件导入（按照官方规范） =====
try:
    from app.core.event import eventmanager, Event, EventType
except Exception:  # 便于在IDE中静态检查，不影响运行环境
    class Dummy:
        def register(self, *a, **k):
            def deco(f):
                return f
            return deco
        def send_event(self, *a, **k):
            pass
    eventmanager = Dummy()
    class EventType:
        NameRecognize = "name.recognize"
        NameRecognizeResult = "name.recognize.result"
    class Event: ...

LOGGER = logging.getLogger(__name__)

# ===================== 配置默认值 =====================
DEFAULT_LLM_BASE = "https://api.gptapi.us/v1"  # 你的网关通过了 /v1/chat/completions + json_object
DEFAULT_LLM_MODEL = "deepseek-v3"
LLM_TIMEOUT = 12  # 秒（留出解析/汇总时间 < 15s 总SLA）
HTTP_TIMEOUT = 3   # 外部数据源单次查询超时

# ===================== 评分权重（可在配置页覆盖） =====================
SCORE_WEIGHTS_DEFAULT = {
    # ID 命中
    "tmdb_hit": 60,
    "douban_hit": 50,
    "bangumi_hit": 55,
    "trakt_hit": 40,
    # 标题相似度（0-1）* 30
    "title_sim_max": 30,
    # 年份匹配（±1 给满分）
    "year_match_max": 15,
    # S/E 匹配
    "se_match_max": 25,
    # 多源一致性奖励（每一致对 + 值）
    "consistency_pair": 12,  # 两两一致每对 +12，上限 36
    # 类型/地区加成
    "anime_bangumi_bonus": 10,
    "movie_imdb_bonus": 10,
    # AI（LLM）贡献
    "ai_structured": 15,
    "ai_field_completeness_max": 15,
    # 惩罚项
    "year_conflict": -30,
    "id_conflict": -50,
    "unstructured_penalty": -20,
}

# ===================== 工具函数 =====================

def _json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(text[i:j+1])
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


def _sim(a: str, b: str) -> float:
    """极简相似度：Jaccard on token set（中/英文都能工作个七八成）。"""
    if not a or not b:
        return 0.0
    at = set(re.split(r"[^\w\u4e00-\u9fa5]+", a.lower())) - {""}
    bt = set(re.split(r"[^\w\u4e00-\u9fa5]+", b.lower())) - {""}
    if not at or not bt:
        return 0.0
    return len(at & bt) / float(len(at | bt))


def _safe_int(x: Any) -> Optional[int]:
    try:
        v = int(x)
        return v if v > 0 else None
    except Exception:
        return None


# ===================== 外部数据源封装（可选，按配置启用） =====================
class Sources:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def search_trakt(self, title: str) -> List[dict]:
        key = self.cfg.get("trakt_api_key")
        if not key:
            return []
        url = "https://api.trakt.tv/search/multi"
        headers = {
            "Content-Type": "application/json",
            "trakt-api-key": key,
            "trakt-api-version": "2",
        }
        try:
            r = requests.get(url, params={"query": title}, headers=headers, timeout=HTTP_TIMEOUT)
            if r.ok:
                return r.json() or []
        except Exception as e:
            LOGGER.debug("trakt search err: %s", e)
        return []

    def search_bangumi(self, title: str) -> List[dict]:
        # 简易调用公开搜索API；可按需改为更丰富的参数
        url = "https://api.bgm.tv/search/subject/{}".format(title)
        try:
            r = requests.get(url, params={"type": 2, "responseGroup": "small"}, timeout=HTTP_TIMEOUT)
            if r.ok:
                return r.json() or []
        except Exception as e:
            LOGGER.debug("bangumi search err: %s", e)
        return []

    def search_tmdb(self, title: str, mtype_hint: str = "") -> List[dict]:
        key = self.cfg.get("tmdb_api_key")
        if not key:
            return []
        base = "https://api.themoviedb.org/3"
        path = "/search/tv" if mtype_hint == "tv" else "/search/movie"
        try:
            r = requests.get(base + path, params={"api_key": key, "query": title}, timeout=HTTP_TIMEOUT)
            if r.ok:
                return r.json().get("results", [])
        except Exception as e:
            LOGGER.debug("tmdb search err: %s", e)
        return []


# ===================== LLM 调用（JSON 模式强制） =====================
class LLM:
    STRICT_PROMPT = (
        "你是严格模式的命名解析器。只输出一个JSON对象，不得包含解释、Markdown或代码块。"
        "字段：name, version, part, year, resolution, season, episode。"
        "规则：year为4位数字或null；season/episode为正整数或null；其余为字符串或null。"
        "无法解析时输出 {}。示例：{\"name\":\"xxx\",\"year\":\"2024\",\"season\":1,\"episode\":2,\"version\":null,\"part\":null,\"resolution\":\"1080p\"}"
    )

    def __init__(self, base: str, key: str, model: str):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model

    def parse(self, title: str) -> dict:
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.STRICT_PROMPT},
                {"role": "user", "content": f"解析以下标题为JSON：{title}"},
            ],
            "temperature": 0,
            "top_p": 0.1,
            "response_format": {"type": "json_object"},
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            content = (data.get("choices", [{}])[0].get("message", {}).get("content", ""))
            js = _json(content) or {}
            return js if isinstance(js, dict) else {}
        except Exception as e:
            LOGGER.warning("LLM parse failed: %s", e)
            return {}


# ===================== 打分器 =====================
@dataclass
class ScoreBreakdown:
    total: int
    items: List[Tuple[str, int]] = field(default_factory=list)

    def add(self, label: str, score: int):
        self.items.append((label, score))
        self.total += score


class Scorer:
    def __init__(self, weights: dict):
        self.w = {**SCORE_WEIGHTS_DEFAULT, **(weights or {})}

    def score(self, title: str, ai: dict, corroborations: dict) -> ScoreBreakdown:
        bd = ScoreBreakdown(total=0)

        # AI 结构化贡献
        if ai:
            bd.add("ai_structured", self.w["ai_structured"])
            # 字段完备度：name/year/season/episode/resolution/version/part
            fields = ["name", "year", "season", "episode", "resolution", "version", "part"]
            filled = sum(1 for f in fields if ai.get(f) not in (None, "", {}))
            comp = int(self.w["ai_field_completeness_max"] * filled / len(fields))
            bd.add("ai_field_completeness", comp)
        else:
            bd.add("unstructured_penalty", self.w["unstructured_penalty"])

        # 单源 ID 命中
        if corroborations.get("tmdb_id"):
            bd.add("tmdb_hit", self.w["tmdb_hit"])
        if corroborations.get("douban_id"):
            bd.add("douban_hit", self.w["douban_hit"])
        if corroborations.get("bangumi_id"):
            bd.add("bangumi_hit", self.w["bangumi_hit"])
        if corroborations.get("trakt_id"):
            bd.add("trakt_hit", self.w["trakt_hit"])

        # 标题相似度（取多源最高相似度）
        name = ai.get("name") or title
        sims = []
        for key in ("tmdb_title", "douban_title", "bangumi_title", "trakt_title"):
            if corroborations.get(key):
                sims.append(_sim(name, corroborations[key]))
        if sims:
            simmax = max(sims)
            bd.add("title_sim", int(simmax * self.w["title_sim_max"]))

        # 年份匹配
        ay = ai.get("year")
        by = corroborations.get("year")  # 聚合后的“可信年份”
        try:
            ay_i = int(ay) if ay else None
            by_i = int(by) if by else None
        except Exception:
            ay_i = by_i = None
        if ay_i and by_i:
            diff = abs(ay_i - by_i)
            if diff == 0:
                bd.add("year_match", self.w["year_match_max"])
            elif diff == 1:
                bd.add("year_match", int(self.w["year_match_max"] * 0.6))
            else:
                bd.add("year_conflict", self.w["year_conflict"])

        # S/E 匹配
        se_ok = corroborations.get("se_match", False)
        if se_ok:
            bd.add("se_match", self.w["se_match_max"])

        # 多源一致性奖励（两两一致对数 * pair 分，上限 36）
        ids = [x for x in (corroborations.get("tmdb_id"), corroborations.get("douban_id"),
                           corroborations.get("bangumi_id"), corroborations.get("trakt_id")) if x]
        pairs = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] == ids[j]:
                    pairs += 1
        bd.add("consistency", min(3, pairs) * self.w["consistency_pair"])  # 上限 3 对

        # 类型/地区加成
        if corroborations.get("is_anime") and corroborations.get("bangumi_id"):
            bd.add("anime_bangumi_bonus", self.w["anime_bangumi_bonus"])
        if corroborations.get("is_movie") and corroborations.get("imdb_id"):
            bd.add("movie_imdb_bonus", self.w["movie_imdb_bonus"])

        return bd


# ===================== 主插件类 =====================
class MultisourceAIRecognizer:
    plugin_name = "多源AI识别与评分"
    plugin_desc = "LLM + Douban/Trakt/Bangumi/TMDB 多源互证，积分制评分（可>100），低分入人工队列并支持自选目录。"
    plugin_icon = "https://raw.githubusercontent.com/jxxghp/MoviePilot-Plugins/main/icons/chatgpt.png"
    plugin_version = "1.0.0"

    # —— 可视化配置（可在插件配置页调整） ——
    _cfg: dict
    _queue: Dict[str, dict]  # 人工确认队列（内存）

    def __init__(self):
        self._cfg = {
            "llm_base": DEFAULT_LLM_BASE,
            "llm_model": DEFAULT_LLM_MODEL,
            "llm_key": "",
            "trakt_api_key": "",
            "tmdb_api_key": "",
            "use_mp_search": True,     # 通过 MP 的 /api/v1/media/search 补强（推荐）
            "mp_api_base": "http://127.0.0.1:3001",
            "mp_api_token": "",
            "threshold_auto": 120,
            "threshold_manual": 80,
            "weights": SCORE_WEIGHTS_DEFAULT,
        }
        self._queue = {}

    # ========== 插件配置页面字段（后端定义，前端自动渲染） ==========
    def get_setting(self) -> Optional[dict]:
        """返回一个基于 Vuetify 的 JSON 配置页（官方支持）。"""
        return {
            "element": "v-container",
            "props": {"fluid": True},
            "children": [
                {"element": "v-row", "children": [
                    {"element": "v-col", "props": {"cols": 12, "md": 6}, "children": [
                        {"element": "v-text-field", "props": {
                            "label": "LLM Base URL", "model": "llm_base",
                            "placeholder": "https://api.gptapi.us/v1"}},
                        {"element": "v-text-field", "props": {
                            "label": "LLM Model", "model": "llm_model", "placeholder": "deepseek-v3"}},
                        {"element": "v-text-field", "props": {
                            "label": "LLM API Key", "model": "llm_key", "type": "password"}},
                        {"element": "v-switch", "props": {
                            "label": "使用 MoviePilot /api/v1/media/search 补强", "model": "use_mp_search"}},
                        {"element": "v-text-field", "props": {
                            "label": "MP API Base", "model": "mp_api_base", "placeholder": "http://127.0.0.1:3001"}},
                        {"element": "v-text-field", "props": {
                            "label": "MP API Token", "model": "mp_api_token", "type": "password"}},
                    ]},
                    {"element": "v-col", "props": {"cols": 12, "md": 6}, "children": [
                        {"element": "v-text-field", "props": {
                            "label": "Trakt API Key", "model": "trakt_api_key"}},
                        {"element": "v-text-field", "props": {
                            "label": "TMDB API Key", "model": "tmdb_api_key"}},
                        {"element": "v-text-field", "props": {
                            "label": "自动通过阈值", "model": "threshold_auto", "type": "number"}},
                        {"element": "v-text-field", "props": {
                            "label": "人工队列阈值（下限）", "model": "threshold_manual", "type": "number"}},
                    ]}
                ]},
                {"element": "v-divider"},
                {"element": "v-alert", "props": {
                    "type": "info", "text": True,
                    "children": [
                        "打分说明：ID命中/相似度/年份/季集/一致性/加成/惩罚 + AI贡献。分数可超 100。",
                        " 自动≥阈值直接返回识别结果；介于两阈值入人工队列；低于下限直接强制人工。",
                    ]
                }}
            ]
        }

    def get_state(self) -> Optional[dict]:
        """返回当前配置（MoviePilot 前端会调用显示/保存）。"""
        return self._cfg

    def set_state(self, state: dict):
        self._cfg.update(state or {})

    # ========== 插件页面（人工队列） ==========
    def get_page(self) -> Optional[dict]:
        # 简单列表：显示队列条目（title/name/year/score）+ 操作按钮
        items = []
        for k, v in list(self._queue.items()):
            items.append({
                "id": k,
                "title": v.get("title"),
                "name": v.get("ai", {}).get("name"),
                "year": v.get("ai", {}).get("year"),
                "score": v.get("score", {}).get("total", 0),
            })
        return {
            "element": "v-container",
            "props": {"fluid": True},
            "children": [
                {"element": "v-data-table", "props": {
                    "items": items,
                    "headers": [
                        {"title": "ID", "value": "id"},
                        {"title": "标题", "value": "title"},
                        {"title": "名称", "value": "name"},
                        {"title": "年份", "value": "year"},
                        {"title": "得分", "value": "score"},
                        {"title": "操作", "value": "actions"},
                    ],
                    "item": {
                        "actions": {
                            "element": "v-btn",
                            "props": {
                                "color": "primary", "text": True,
                                "onclick": {"api": "/confirm", "method": "POST", "json": {"id": "{{item.id}}"}}
                            },
                            "children": ["确认并整理（选择目录）"]
                        }
                    }
                }}
            ]
        }

    # ========== 暴露 API：确认/目录浏览 ==========
    def get_api(self) -> Optional[List[dict]]:
        return [
            {
                "path": "/queue",
                "endpoint": self.api_queue,
                "methods": ["GET"],
                "summary": "获取人工队列",
                "description": "返回待确认条目列表",
            },
            {
                "path": "/confirm",
                "endpoint": self.api_confirm,
                "methods": ["POST"],
                "summary": "确认某条并执行转移/整理",
                "description": "根据ID确认，前端会先弹出目录选择再提交",
            },
            {
                "path": "/storage_list",
                "endpoint": self.api_storage_list,
                "methods": ["POST"],
                "summary": "浏览目录",
                "description": "代理调用 /api/v1/storage/list",
            },
            {
                "path": "/storage_mkdir",
                "endpoint": self.api_storage_mkdir,
                "methods": ["POST"],
                "summary": "新建目录",
                "description": "代理调用 /api/v1/storage/mkdir",
            },
        ]

    # ========== API 实现 ==========
    def _mp_headers(self) -> dict:
        token = self._cfg.get("mp_api_token")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def api_queue(self):
        return {"data": list(self._queue.values())}

    def api_storage_list(self, path: str = "/", storage: Optional[str] = None):
        url = f"{self._cfg['mp_api_base'].rstrip('/')}/api/v1/storage/list"
        r = requests.post(url, json={"path": path, "storage": storage}, headers=self._mp_headers(), timeout=HTTP_TIMEOUT)
        return r.json()

    def api_storage_mkdir(self, path: str, storage: Optional[str] = None):
        url = f"{self._cfg['mp_api_base'].rstrip('/')}/api/v1/storage/mkdir"
        r = requests.post(url, json={"path": path, "storage": storage}, headers=self._mp_headers(), timeout=HTTP_TIMEOUT)
        return r.json()

    def api_confirm(self, id: str, target_storage: str = "", target_path: str = "", background: bool = True):
        item = self._queue.get(id)
        if not item:
            return {"code": 404, "msg": "not found"}
        # 调用 transfer/manual 完成入库（可按需扩展 episode_format 等）
        url = f"{self._cfg['mp_api_base'].rstrip('/')}/api/v1/transfer/manual"
        body = {
            "items": [
                {
                    "path": item.get("src_path", ""),
                    "type": "file",
                    "target_storage": target_storage,
                    "target_path": target_path,
                    "mediainfo": {
                        "name": item.get("ai", {}).get("name"),
                        "year": item.get("ai", {}).get("year"),
                        "season": _safe_int(item.get("ai", {}).get("season")),
                        "episode": _safe_int(item.get("ai", {}).get("episode")),
                    }
                }
            ],
            "background": background
        }
        r = requests.post(url, json=body, headers=self._mp_headers(), timeout=HTTP_TIMEOUT)
        # 确认后从队列移除
        self._queue.pop(id, None)
        return r.json()

    # ========== 识别链：NameRecognize ==========
    @eventmanager.register(EventType.NameRecognize)
    def on_name_recognize(self, event: Event):
        """主程序无法识别时触发，要求在 ~15s 内返回结果或空结果。"""
        st = time.time()
        data = getattr(event, "event_data", {}) or {}
        title = data.get("title") or data.get("name") or ""
        subtitle = data.get("subtitle", "")
        src_path = data.get("path", "")

        if not self._cfg.get("llm_key") or not title:
            # 立即空结果返回，避免阻塞
            eventmanager.send_event(EventType.NameRecognizeResult, {"title": title})
            return

        # 1) LLM 严格 JSON 解析
        llm = LLM(self._cfg["llm_base"], self._cfg["llm_key"], self._cfg["llm_model"])
        ai = llm.parse(title)

        # 2) 外部与 MP 搜索互证（按需并行/串行，这里串行+短超时）
        sources = Sources(self._cfg)
        corroborations: Dict[str, Any] = {}

        # MP 的 /api/v1/media/search（聚合了系统内置源），常作为“可信年份/别名”来源
        if self._cfg.get("use_mp_search"):
            try:
                url = f"{self._cfg['mp_api_base'].rstrip('/')}/api/v1/media/search"
                r = requests.get(url, params={"title": ai.get("name") or title, "type": "media"}, headers=self._mp_headers(), timeout=HTTP_TIMEOUT)
                if r.ok:
                    res = r.json() or {}
                    # 简化处理：取第一候选
                    cand = (res.get("data") or [{}])[0] if isinstance(res.get("data"), list) else {}
                    corroborations["year"] = cand.get("year") or corroborations.get("year")
                    if cand.get("tmdbid"):
                        corroborations["tmdb_id"] = cand.get("tmdbid")
                    if cand.get("name"):
                        corroborations["douban_title"] = cand.get("name")  # 作为别名来源之一
            except Exception as e:
                LOGGER.debug("mp search err: %s", e)

        # Trakt
        try:
            tks = sources.search_trakt(ai.get("name") or title)
            if tks:
                t0 = tks[0]
                corroborations["trakt_id"] = (t0.get("show") or t0.get("movie") or {}).get("ids", {}).get("slug") or t0.get("score")
                nm = (t0.get("show") or t0.get("movie") or {}).get("title")
                if nm:
                    corroborations["trakt_title"] = nm
        except Exception:
            pass

        # Bangumi（动漫）
        try:
            bgs = sources.search_bangumi(ai.get("name") or title)
            if bgs:
                b0 = bgs[0]
                corroborations["bangumi_id"] = b0.get("id")
                corroborations["bangumi_title"] = b0.get("name")
                corroborations["is_anime"] = True
        except Exception:
            pass

        # TMDB（可选）
        try:
            tmdbs = sources.search_tmdb(ai.get("name") or title, "tv" if _safe_int(ai.get("season")) else "movie")
            if tmdbs:
                m0 = tmdbs[0]
                corroborations["tmdb_id"] = corroborations.get("tmdb_id") or m0.get("id")
                corroborations["tmdb_title"] = m0.get("name") or m0.get("title")
                if not corroborations.get("year"):
                    date = m0.get("first_air_date") or m0.get("release_date") or ""
                    if date[:4].isdigit():
                        corroborations["year"] = date[:4]
                corroborations["is_movie"] = bool(m0.get("title"))
        except Exception:
            pass

        # S/E 匹配（基于标题粗提 + AI 字段）
        se_title = re.search(r"S(\d+)[Eex](\d+)", title, re.I)
        s_ai, e_ai = _safe_int(ai.get("season")), _safe_int(ai.get("episode"))
        corroborations["se_match"] = bool(se_title and s_ai == _safe_int(se_title.group(1)) and e_ai == _safe_int(se_title.group(2)))

        # 3) 评分
        scorer = Scorer(self._cfg.get("weights") or {})
        bd = scorer.score(title, ai, corroborations)

        # 4) 路由：≥auto 直接返回；manual~auto 入人工队列；<manual 直接空结果
        auto_th = int(self._cfg.get("threshold_auto", 120))
        man_th = int(self._cfg.get("threshold_manual", 80))

        # 确保在 15 秒内返回
        if time.time() - st > 14.0:
            LOGGER.warning("timeout guard: returning empty")
            eventmanager.send_event(EventType.NameRecognizeResult, {"title": title})
            return

        if bd.total >= auto_th:
            # 返回识别结果（最小字段集即可）
            result = {
                "title": title,
                "name": ai.get("name") or title,
                "year": ai.get("year"),
                "season": _safe_int(ai.get("season")),
                "episode": _safe_int(ai.get("episode")),
            }
            eventmanager.send_event(EventType.NameRecognizeResult, result)
            return

        if man_th <= bd.total < auto_th:
            # 入人工队列 + 空结果返回
            qid = str(uuid.uuid4())
            self._queue[qid] = {
                "id": qid,
                "title": title,
                "src_path": src_path,
                "ai": ai,
                "corroborations": corroborations,
                "score": {"total": bd.total, "items": bd.items},
                "ts": time.time(),
            }
            eventmanager.send_event(EventType.NameRecognizeResult, {"title": title})
            return

        # 强制人工：空结果
        eventmanager.send_event(EventType.NameRecognizeResult, {"title": title})
