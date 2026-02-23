# -*- coding: utf-8 -*-
"""
AI 辅助识别插件 for MoviePilot v2.0.0
- LLM (JSON-only) 辅助解析媒体标题，支持多服务商预设
- 积分制评分（可>100），阈值可配
- 智能触发AI：仅当主程序识别结果不完整时才问AI；也支持"总是/手动"
- LLM 结果缓存、失败重试、可配超时、自定义 System Prompt
- 队列管理：删除/清空/容量上限
- 通知：MoviePilot 内置 + 独立 Telegram Bot 频道
- 识别统计（调用/成功/失败/成功率）
- 仪表板、按钮交互、工作流、消息回调、配置页、自检
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import threading
import random
import string
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests

# ===== MoviePilot 基础导入 =====
from app.core.event import eventmanager, Event, EventType, ChainEventType
try:
    from app.core.plugin import PluginBase as MPPluginBase
except Exception:
    try:
        from app.core.plugin import Plugin as MPPluginBase
    except Exception:
        try:
            from app.plugins import Plugin as MPPluginBase
        except Exception as e:
            raise ImportError(
                "无法导入插件基类，请升级 MoviePilot。"
            ) from e
from app.log import logger
from app.core.config import settings


# ========= 全局常量 =========
LLM_SYSTEM_PROMPT_DEFAULT = (
    "你是严格模式的命名解析器。只输出一个JSON对象，不得包含解释、Markdown或代码块。"
    "字段：name, version, part, year, resolution, season, episode。"
    "规则：year为4位数字或null；season/episode为正整数或null；其余为字符串或null。"
    '无法解析时输出{}。示例：'
    '{"name":"xxx","year":"2024","season":1,"episode":2,'
    '"version":null,"part":null,"resolution":"1080p"}'
)

SCORE_WEIGHTS_DEFAULT = {
    "tmdb_hit": 60, "douban_hit": 50, "bangumi_hit": 55, "trakt_hit": 40,
    "title_sim_max": 30, "year_match_max": 15, "se_match_max": 25,
    "consistency_pair": 12,
    "anime_bangumi_bonus": 10, "movie_imdb_bonus": 10,
    "ai_structured": 15, "ai_field_completeness_max": 15,
    "year_conflict": -30, "id_conflict": -50, "unstructured_penalty": -20,
}
THRESHOLD_AUTO_DEFAULT = 120
THRESHOLD_MANUAL_DEFAULT = 80

LLM_PROVIDERS = {
    "deepseek": {"name": "DeepSeek", "base": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openai": {"name": "OpenAI (ChatGPT)", "base": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "gemini": {"name": "Google Gemini", "base": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.0-flash"},
    "qwen": {"name": "通义千问 (Qwen)", "base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "zhipu": {"name": "智谱 (GLM)", "base": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    "moonshot": {"name": "月之暗面 (Kimi)", "base": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "yi": {"name": "零一万物 (Yi)", "base": "https://api.lingyiwanwu.com/v1", "model": "yi-lightning"},
    "groq": {"name": "Groq", "base": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "custom": {"name": "自定义", "base": "", "model": ""},
}
LLM_PROVIDER_OPTIONS = [{"title": v["name"], "value": k} for k, v in LLM_PROVIDERS.items()]

_STATS_DEFAULT: Dict[str, int] = {"total_calls": 0, "success": 0, "fail": 0}


# ========= 小工具 =========
def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _gen_id(n=8):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def _title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ========= LLM 调用（JSON 模式 + 重试）=========
class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = 30):
        self.base = (base_url or "").rstrip("/")
        self.key = api_key or ""
        self.model = model or "deepseek-chat"
        self.timeout = timeout

    def parse_title(self, title: str, system_prompt: str,
                    max_retries: int = 0) -> Optional[Dict[str, Any]]:
        if not self.base or not self.key:
            return None
        url = f"{self.base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"解析以下标题为JSON：{title}"},
            ],
            "temperature": 0,
            "top_p": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload,
                                  timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                content = (data.get("choices", [{}])[0]
                           .get("message", {}).get("content", ""))
                return self._robust_json(content)
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 4))
        logger.warning(f"[MSAIR][LLM] failed after {max_retries + 1} "
                       f"attempt(s): {last_err}")
        return None

    @staticmethod
    def _robust_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i: j + 1])
            except Exception:
                pass
        try:
            return json.loads(text)
        except Exception:
            return None


# ========= MP 后端直连 =========
class MPClient:
    def __init__(self, base: str = "", token: str = "", timeout: int = 6):
        self.base = base.rstrip("/") if base else ""
        self.token = token or ""
        self.timeout = timeout

    def _headers(self):
        if self.token:
            return {"Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"}
        return {"Content-Type": "application/json"}

    def search(self, title: str,
               mtype: str = "media") -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.get(f"{self.base}/api/v1/media/search",
                             params={"title": title, "type": mtype},
                             headers=self._headers(), timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug(f"[MSAIR][MP] search error: {e}")
        return None

    def transfer_manual(self,
                        body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.post(f"{self.base}/api/v1/transfer/manual",
                              headers=self._headers(), json=body,
                              timeout=self.timeout)
            return r.json()
        except Exception as e:
            logger.debug(f"[MSAIR][MP] transfer_manual error: {e}")
            return None

    def download_with_media(self,
                            body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.post(f"{self.base}/api/v1/download/",
                              headers=self._headers(), json=body,
                              timeout=self.timeout)
            return r.json()
        except Exception as e:
            logger.debug(f"[MSAIR][MP] download error: {e}")
            return None


# ========= 打分器 =========
class ScoreBreakdown:
    def __init__(self):
        self.items: List[Tuple[str, int]] = []
        self.total: int = 0

    def add(self, key: str, val: int):
        self.items.append((key, int(val)))
        self.total += int(val)


class Scorer:
    def __init__(self, weights: Dict[str, int]):
        self.w = {**SCORE_WEIGHTS_DEFAULT, **(weights or {})}

    def score(self, ai: Dict[str, Any],
              ext: Dict[str, Any]) -> ScoreBreakdown:
        bd = ScoreBreakdown()
        name = (ai or {}).get("name") or ""
        year = (ai or {}).get("year")
        season = _safe_int((ai or {}).get("season"))
        episode = _safe_int((ai or {}).get("episode"))

        if ai and isinstance(ai, dict) and ai.get("name"):
            bd.add("ai_structured", self.w["ai_structured"])
            fields = ["name", "year", "season", "episode",
                      "resolution", "version", "part"]
            filled = sum(1 for f in fields
                         if ai.get(f) not in (None, "", []))
            bd.add("ai_field_completeness",
                   int(round(self.w["ai_field_completeness_max"]
                             * (filled / len(fields)))))
        elif ai is not None:
            bd.add("unstructured_penalty", self.w["unstructured_penalty"])

        if ext.get("tmdbid"):
            bd.add("tmdb_hit", self.w["tmdb_hit"])
        if ext.get("doubanid"):
            bd.add("douban_hit", self.w["douban_hit"])
        if ext.get("bangumiid"):
            bd.add("bangumi_hit", self.w["bangumi_hit"])
        if ext.get("traktid"):
            bd.add("trakt_hit", self.w["trakt_hit"])
        if ext.get("id_conflict"):
            bd.add("id_conflict", self.w["id_conflict"])

        names = ext.get("names") or []
        if name and names:
            sims = [_title_similarity(name, n) for n in names]
            bd.add("title_sim",
                   int(round(max(sims) * self.w["title_sim_max"])))

        y_mp = ext.get("year")
        if year and y_mp:
            try:
                if str(year) == str(y_mp):
                    bd.add("year_match", self.w["year_match_max"])
                elif abs(int(year) - int(y_mp)) == 1:
                    bd.add("year_match_near",
                           int(round(self.w["year_match_max"] * 0.6)))
                else:
                    bd.add("year_conflict", self.w["year_conflict"])
            except Exception:
                pass

        se_mp = ext.get("se") or {}
        if season is not None and se_mp.get("season") == season:
            bd.add("season_match",
                   int(round(self.w["se_match_max"] * 0.6)))
        if episode is not None and se_mp.get("episode") == episode:
            bd.add("episode_match",
                   int(round(self.w["se_match_max"] * 0.4)))

        if _safe_int(ext.get("agree_pairs")):
            bd.add("consistency_bonus",
                   self.w["consistency_pair"]
                   * min(3, int(ext["agree_pairs"])))

        if ext.get("is_anime") and ext.get("bangumiid"):
            bd.add("anime_bangumi_bonus", self.w["anime_bangumi_bonus"])
        if (ext.get("is_movie") and ext.get("imdbid")
                and ext.get("tmdbid")):
            bd.add("movie_imdb_bonus", self.w["movie_imdb_bonus"])

        return bd


# ========= 插件主体 =========
class Multisource_Ai_Recognizer(MPPluginBase):
    plugin_name = "AI辅助识别与评分"
    plugin_desc = (
        "LLM 辅助媒体标题解析；积分制评分；缓存/重试/统计；"
        "队列管理；MoviePilot+Telegram 双通道通知"
    )
    plugin_icon = ("https://raw.githubusercontent.com/jxxghp/"
                   "MoviePilot-Plugins/main/icons/chatgpt.png")
    plugin_version = "2.0.0"
    plugin_author = "maoxiongnet"
    plugin_order = 50

    # ═══════ 生命周期 ═══════

    def __init__(self):
        super().__init__()
        self._cfg: Dict[str, Any] = {}
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._selftest: Dict[str, Any] = {}
        self._llm_cache: Dict[str, Tuple[float, Optional[Dict]]] = {}
        self._stats: Dict[str, int] = {**_STATS_DEFAULT}

    def init_plugin(self, config: dict = None):
        default_cfg: Dict[str, Any] = {
            "llm_provider": "deepseek",
            "llm_base": "",
            "llm_model": "",
            "llm_key": "",
            "llm_timeout": 30,
            "llm_retry": 2,
            "llm_system_prompt": "",
            "cache_ttl": 3600,
            "queue_max_size": 500,
            "mp_api_base": "",
            "mp_api_token": "",
            "ask_mode": "smart",
            "auto_download": False,
            "threshold_auto": THRESHOLD_AUTO_DEFAULT,
            "threshold_manual": THRESHOLD_MANUAL_DEFAULT,
            "weights": SCORE_WEIGHTS_DEFAULT,
            "tg_bot_token": "",
            "tg_chat_id": "",
        }
        if config:
            default_cfg.update(config)
            saved_q = config.get("_queue")
            if isinstance(saved_q, dict):
                with self._lock:
                    self._queue = saved_q
            saved_s = config.get("_stats")
            if isinstance(saved_s, dict):
                with self._lock:
                    self._stats = {**_STATS_DEFAULT, **saved_s}
        self._cfg = default_cfg

    def stop_service(self):
        self._llm_cache.clear()
        self._persist()
        logger.info("[MSAIR] plugin stopped, cache cleared, state saved")

    # ═══════ 内部工具 ═══════

    def _get_system_prompt(self) -> str:
        custom = (self._cfg.get("llm_system_prompt") or "").strip()
        return custom if custom else LLM_SYSTEM_PROMPT_DEFAULT

    def _make_llm_client(self) -> LLMClient:
        provider = self._cfg.get("llm_provider", "custom")
        preset = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["custom"])
        base = self._cfg.get("llm_base") or preset.get("base", "")
        model = self._cfg.get("llm_model") or preset.get("model", "")
        key = self._cfg.get("llm_key", "")
        timeout = _safe_int(self._cfg.get("llm_timeout"), 30)
        return LLMClient(base, key, model, timeout=timeout)

    def _llm_parse_cached(self, title: str) -> Optional[Dict[str, Any]]:
        """LLM 解析（缓存 + 重试 + 统计）"""
        cache_key = hashlib.md5(
            title.strip().lower().encode()).hexdigest()
        ttl = _safe_int(self._cfg.get("cache_ttl"), 3600)
        now = time.time()

        if ttl > 0 and cache_key in self._llm_cache:
            ts, cached = self._llm_cache[cache_key]
            if now - ts < ttl:
                logger.debug(f"[MSAIR] cache hit: {title}")
                return cached

        llm = self._make_llm_client()
        retries = _safe_int(self._cfg.get("llm_retry"), 2)
        result = llm.parse_title(title, self._get_system_prompt(),
                                 max_retries=retries)

        if ttl > 0:
            self._llm_cache[cache_key] = (now, result)

        ok = result is not None and bool((result or {}).get("name"))
        self._update_stats(ok)
        return result

    def _add_to_queue(self, iid: str, item: dict):
        max_size = _safe_int(self._cfg.get("queue_max_size"), 500)
        with self._lock:
            self._queue[iid] = item
            if max_size > 0:
                while len(self._queue) > max_size:
                    oldest = next(iter(self._queue))
                    if oldest == iid:
                        break
                    del self._queue[oldest]

    def _update_stats(self, success: bool):
        with self._lock:
            self._stats["total_calls"] = \
                self._stats.get("total_calls", 0) + 1
            if success:
                self._stats["success"] = \
                    self._stats.get("success", 0) + 1
            else:
                self._stats["fail"] = \
                    self._stats.get("fail", 0) + 1

    def _persist(self):
        try:
            with self._lock:
                qs = copy.deepcopy(self._queue)
                ss = copy.deepcopy(self._stats)
            self.update_config({**self._cfg, "_queue": qs, "_stats": ss})
        except Exception as e:
            logger.debug(f"[MSAIR] persist failed: {e}")

    _save_queue = _persist

    # ═══════ 通知 ═══════

    def _notify(self, title: str, text: str):
        try:
            self.post_message(title=title, text=text)
        except Exception as e:
            logger.debug(f"[MSAIR] post_message failed: {e}")
        self._notify_telegram(title, text)

    def _notify_telegram(self, title: str, text: str):
        token = (self._cfg.get("tg_bot_token") or "").strip()
        chat_id = (self._cfg.get("tg_chat_id") or "").strip()
        if not token or not chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={
                "chat_id": chat_id,
                "text": f"<b>{title}</b>\n{text}",
                "parse_mode": "HTML",
            }, timeout=10)
        except Exception as e:
            logger.debug(f"[MSAIR] telegram notify failed: {e}")

    @staticmethod
    def _build_ext(data) -> Dict[str, Any]:
        ext: Dict[str, Any] = {}
        for attr in ("tmdbid", "doubanid", "bangumiid",
                     "traktid", "imdbid"):
            val = getattr(data, attr, None)
            if val:
                ext[attr] = val
        existing_name = getattr(data, "name", None)
        if existing_name:
            ext["names"] = [existing_name]
        existing_year = getattr(data, "year", None)
        if existing_year:
            ext["year"] = str(existing_year)
        s = _safe_int(getattr(data, "season", None))
        ep = _safe_int(getattr(data, "episode", None))
        if s is not None or ep is not None:
            ext["se"] = {}
            if s is not None:
                ext["se"]["season"] = s
            if ep is not None:
                ext["se"]["episode"] = ep
        mtype = (getattr(data, "type", None)
                 or getattr(data, "media_type", None))
        if mtype:
            ms = str(mtype).lower()
            if "anime" in ms or "动漫" in ms:
                ext["is_anime"] = True
            if "movie" in ms or "电影" in ms:
                ext["is_movie"] = True
        return ext

    # ═══════ 配置页 ═══════

    def get_setting(self) -> Optional[dict]:
        return {
            "element": "v-container",
            "props": {"fluid": True},
            "children": [
                # Row 1: LLM 基础 + 行为
                {"element": "v-row", "children": [
                    {"element": "v-col", "props": {"cols": 12, "md": 6},
                     "children": [
                         {"element": "v-select", "props": {
                             "label": "LLM 服务商", "model": "llm_provider",
                             "items": LLM_PROVIDER_OPTIONS,
                             "hint": "选择后自动填充 Base URL 和 Model",
                             "persistent-hint": True}},
                         {"element": "v-text-field", "props": {
                             "label": "LLM API Key", "model": "llm_key",
                             "type": "password"}},
                         {"element": "v-text-field", "props": {
                             "label": "Base URL（留空使用服务商预设）",
                             "model": "llm_base",
                             "placeholder": "留空 = 使用预设地址"}},
                         {"element": "v-text-field", "props": {
                             "label": "Model（留空使用服务商预设）",
                             "model": "llm_model",
                             "placeholder": "留空 = 使用预设模型"}},
                     ]},
                    {"element": "v-col", "props": {"cols": 12, "md": 6},
                     "children": [
                         {"element": "v-select", "props": {
                             "label": "AI触发模式", "model": "ask_mode",
                             "items": [
                                 {"title": "智能(smart)", "value": "smart"},
                                 {"title": "总是(always)", "value": "always"},
                                 {"title": "手动(manual)", "value": "manual"},
                             ]}},
                         {"element": "v-switch", "props": {
                             "label": "自动下载（>=自动阈值）",
                             "model": "auto_download"}},
                         {"element": "v-text-field", "props": {
                             "label": "自动通过阈值",
                             "model": "threshold_auto", "type": "number"}},
                         {"element": "v-text-field", "props": {
                             "label": "人工队列阈值（下限）",
                             "model": "threshold_manual", "type": "number"}},
                     ]},
                ]},
                # Row 2: 高级 + 通知
                {"element": "v-row", "children": [
                    {"element": "v-col", "props": {"cols": 12, "md": 6},
                     "children": [
                         {"element": "v-text-field", "props": {
                             "label": "LLM 超时（秒）",
                             "model": "llm_timeout", "type": "number",
                             "hint": "默认 30", "persistent-hint": True}},
                         {"element": "v-text-field", "props": {
                             "label": "LLM 失败重试次数",
                             "model": "llm_retry", "type": "number",
                             "hint": "0=不重试，默认 2",
                             "persistent-hint": True}},
                         {"element": "v-text-field", "props": {
                             "label": "缓存有效期（秒）",
                             "model": "cache_ttl", "type": "number",
                             "hint": "0=禁用缓存，默认 3600",
                             "persistent-hint": True}},
                         {"element": "v-text-field", "props": {
                             "label": "队列容量上限",
                             "model": "queue_max_size", "type": "number",
                             "hint": "0=不限，默认 500",
                             "persistent-hint": True}},
                     ]},
                    {"element": "v-col", "props": {"cols": 12, "md": 6},
                     "children": [
                         {"element": "v-text-field", "props": {
                             "label": "MoviePilot API 地址（可选）",
                             "model": "mp_api_base",
                             "placeholder": "http://localhost:3000"}},
                         {"element": "v-text-field", "props": {
                             "label": "MoviePilot API Token（可选）",
                             "model": "mp_api_token", "type": "password"}},
                         {"element": "v-text-field", "props": {
                             "label": "Telegram Bot Token（独立通知）",
                             "model": "tg_bot_token", "type": "password",
                             "hint": "留空则不发 Telegram",
                             "persistent-hint": True}},
                         {"element": "v-text-field", "props": {
                             "label": "Telegram Chat ID",
                             "model": "tg_chat_id",
                             "hint": "群组/频道/个人 Chat ID",
                             "persistent-hint": True}},
                     ]},
                ]},
                # Row 3: System Prompt
                {"element": "v-row", "children": [
                    {"element": "v-col", "props": {"cols": 12},
                     "children": [
                         {"element": "v-textarea", "props": {
                             "label": "自定义 System Prompt（留空使用默认）",
                             "model": "llm_system_prompt",
                             "rows": 3, "auto-grow": True,
                             "placeholder": LLM_SYSTEM_PROMPT_DEFAULT}},
                     ]},
                ]},
                # Info
                {"element": "v-alert",
                 "props": {"type": "info", "text": True},
                 "children": [
                     "打分：AI贡献/标题相似度/年份/季集匹配；分数可>100。"
                     "推荐阈值：自动>=120；人工80-119；<80交由主程序处理。"
                 ]},
            ],
        }

    def get_state(self) -> Optional[dict]:
        with self._lock:
            return {**self._cfg,
                    "_queue": copy.deepcopy(self._queue),
                    "_stats": copy.deepcopy(self._stats)}

    def set_state(self, state: dict):
        if not state:
            return
        saved_q = state.get("_queue")
        saved_s = state.get("_stats")
        cfg = {k: v for k, v in state.items()
               if k not in ("_queue", "_stats")}
        self._cfg.update(cfg)
        if isinstance(saved_q, dict):
            with self._lock:
                self._queue = saved_q
        if isinstance(saved_s, dict):
            with self._lock:
                self._stats = {**_STATS_DEFAULT, **saved_s}

    # ═══════ NameRecognize 事件 ═══════

    @eventmanager.register(ChainEventType.NameRecognize)
    def on_name_recognize(self, event: Event):
        if not self.is_enabled:
            return
        data = getattr(event, "event_data", None)
        if not data:
            return
        title: str = getattr(data, "title", "") or ""
        if not title.strip():
            return

        mode = self._cfg.get("ask_mode", "smart")
        if mode == "manual":
            return
        if mode == "smart" and getattr(data, "name", None):
            logger.debug(f"[MSAIR] smart skip: '{data.name}'")
            return

        ai = self._llm_parse_cached(title)
        if not ai or not ai.get("name"):
            logger.debug(f"[MSAIR] LLM empty for: {title}")
            return

        ai_name = ai.get("name")
        ai_year = ai.get("year")
        ai_season = _safe_int(ai.get("season"))
        ai_episode = _safe_int(ai.get("episode"))
        if ai_name:
            data.name = str(ai_name)
        if ai_year:
            data.year = str(ai_year)
        if ai_season is not None:
            data.season = ai_season
        if ai_episode is not None:
            data.episode = ai_episode

        logger.info(f"[MSAIR] AI: name={ai_name}, year={ai_year}, "
                     f"S{ai_season}E{ai_episode} | {title}")

        ext = self._build_ext(data)
        scorer = Scorer(self._cfg.get("weights") or {})
        bd = scorer.score(ai, ext)
        total = bd.total
        th_auto = _safe_int(self._cfg.get("threshold_auto"),
                            THRESHOLD_AUTO_DEFAULT)
        th_manual = _safe_int(self._cfg.get("threshold_manual"),
                              THRESHOLD_MANUAL_DEFAULT)

        if total >= th_auto and self._cfg.get("auto_download"):
            body = {"mediainfo": {
                "title": ai_name or title, "year": ai_year,
                "season": ai_season, "episode": ai_episode}}
            if (self._cfg.get("mp_api_base")
                    and self._cfg.get("mp_api_token")):
                resp = MPClient(self._cfg["mp_api_base"],
                                self._cfg["mp_api_token"]
                                ).download_with_media(body)
                logger.info(f"[MSAIR] auto download resp: {resp}")
            else:
                iid = _gen_id()
                self._add_to_queue(iid, {
                    "id": iid, "title": title, "ai": ai, "ext": ext,
                    "score": {"total": total, "items": bd.items},
                    "auto_download_payload": body,
                    "ts": int(time.time())})
                self._persist()
            return

        if total >= th_manual:
            iid = _gen_id()
            self._add_to_queue(iid, {
                "id": iid, "title": title, "ai": ai, "ext": ext,
                "score": {"total": total, "items": bd.items},
                "ts": int(time.time())})
            self._persist()
            se_str = ""
            if ai_season is not None:
                se_str += f" S{ai_season}"
            if ai_episode is not None:
                se_str += f"E{ai_episode}"
            self._notify(
                "AI辅助识别 - 新入队列",
                f"标题: {title}\n"
                f"识别: {ai_name or '?'}"
                f" ({ai_year or '?'}){se_str}\n"
                f"评分: {total}")

    # ═══════ API Token 校验 ═══════

    @staticmethod
    def _check_api_token(apikey: str = "") -> Optional[dict]:
        if apikey != settings.API_TOKEN:
            return {"code": 403, "msg": "API Token 验证失败"}
        return None

    # ═══════ 管理页面 ═══════

    def get_page(self) -> Optional[dict]:
        with self._lock:
            rows = [
                {"id": k, "title": v.get("title"),
                 "name": (v.get("ai") or {}).get("name"),
                 "year": (v.get("ai") or {}).get("year"),
                 "score": (v.get("score") or {}).get("total", 0)}
                for k, v in self._queue.items()
            ]
            q_cnt = len(self._queue)
            stats = {**self._stats}

        tc = stats.get("total_calls", 0)
        succ = stats.get("success", 0)
        fail = stats.get("fail", 0)
        rate = round(succ / tc * 100, 1) if tc > 0 else 0

        stats_card = {
            "element": "v-alert",
            "props": {"type": "info", "text": True, "class": "mb-4"},
            "children": [
                f"LLM 调用: {tc} | 成功: {succ} | 失败: {fail} | "
                f"成功率: {rate}% | 队列: {q_cnt} 条 | "
                f"缓存: {len(self._llm_cache)} 条"
            ]}

        top_btns = {
            "element": "v-row", "children": [
                {"element": "v-col",
                 "props": {"cols": 12, "md": 8},
                 "children": [
                     {"element": "v-btn",
                      "props": {"color": "primary", "class": "mr-2"},
                      "events": {"click": {
                          "api": "plugin/multisource_ai_recognizer/ai_batch",
                          "method": "post",
                          "json": {"scope": "all"}}},
                      "children": ["AI识别（全部）"]},
                     {"element": "v-btn",
                      "props": {"color": "primary", "class": "mr-2"},
                      "events": {"click": {
                          "api": "plugin/multisource_ai_recognizer/ai_batch",
                          "method": "post",
                          "json": {"scope": "selected",
                                   "ids": "{{selectedIds}}"}}},
                      "children": ["AI识别（所选）"]},
                     {"element": "v-btn",
                      "props": {"color": "error", "class": "mr-2"},
                      "events": {"click": {
                          "api": "plugin/multisource_ai_recognizer/queue_clear",
                          "method": "post"}},
                      "children": ["清空队列"]},
                     {"element": "v-btn",
                      "props": {"color": "warning", "class": "mr-2"},
                      "events": {"click": {
                          "api": "plugin/multisource_ai_recognizer/stats_reset",
                          "method": "post"}},
                      "children": ["重置统计"]},
                     {"element": "v-btn",
                      "props": {"color": "secondary"},
                      "events": {"click": {
                          "api": "plugin/multisource_ai_recognizer/selftest",
                          "method": "post"}},
                      "children": ["自检"]},
                 ]},
                {"element": "v-col",
                 "props": {"cols": 12, "md": 4},
                 "children": [
                     {"element": "v-select", "props": {
                         "label": "AI触发模式", "model": "ask_mode",
                         "items": [
                             {"title": "智能(smart)", "value": "smart"},
                             {"title": "总是(always)", "value": "always"},
                             {"title": "手动(manual)", "value": "manual"}],
                         "value": self._cfg.get("ask_mode", "smart")},
                      "events": {"change": {
                          "api": "plugin/multisource_ai_recognizer/config",
                          "method": "post",
                          "json": {"ask_mode": "{{ask_mode}}"}}}}
                 ]},
            ]}

        target_inputs = {
            "element": "v-row", "children": [
                {"element": "v-col", "props": {"cols": 12, "md": 4},
                 "children": [{"element": "v-text-field", "props": {
                     "label": "目标存储类型", "model": "target_storage",
                     "placeholder": "local/nas/..."}}]},
                {"element": "v-col", "props": {"cols": 12, "md": 8},
                 "children": [{"element": "v-text-field", "props": {
                     "label": "目标目录路径", "model": "target_path",
                     "placeholder": "/media/Movies"}}]},
            ]}

        table = {
            "element": "v-data-table",
            "props": {
                "items": rows, "show-select": True,
                "model": "selectedIds",
                "headers": [
                    {"title": "ID", "value": "id"},
                    {"title": "标题", "value": "title"},
                    {"title": "名称", "value": "name"},
                    {"title": "年份", "value": "year"},
                    {"title": "得分", "value": "score"},
                    {"title": "操作", "value": "actions"}],
                "item": {"actions": {
                    "element": "div", "children": [
                        {"element": "v-btn",
                         "props": {"text": True, "color": "primary",
                                   "size": "small", "class": "mr-1"},
                         "events": {"click": {
                             "api": "plugin/multisource_ai_recognizer/confirm",
                             "method": "post",
                             "json": {"id": "{{item.id}}",
                                      "target_storage": "{{target_storage}}",
                                      "target_path": "{{target_path}}"}}},
                         "children": ["确认"]},
                        {"element": "v-btn",
                         "props": {"text": True, "color": "error",
                                   "size": "small"},
                         "events": {"click": {
                             "api": "plugin/multisource_ai_recognizer/queue_delete",
                             "method": "post",
                             "json": {"id": "{{item.id}}"}}},
                         "children": ["删除"]}]}}}}

        children = [stats_card, top_btns, target_inputs, table]

        if self._selftest:
            ok_flag = bool(self._selftest.get("ok"))
            logs_list = self._selftest.get("logs") or []
            txt = "\n".join(logs_list) if logs_list else "（无日志）"
            children.append({
                "element": "v-alert",
                "props": {"type": "success" if ok_flag else "error",
                          "text": True},
                "children": [
                    "自检结果：" + ("通过" if ok_flag else "存在问题")]})
            children.append({
                "element": "v-card", "children": [
                    {"element": "v-card-title",
                     "children": ["自检日志"]},
                    {"element": "v-card-text",
                     "children": [txt]}]})

        return {"element": "v-container", "props": {"fluid": True},
                "children": children}

    # ═══════ API 端点 ═══════

    def get_api(self) -> Optional[List[dict]]:
        return [
            {"path": "/queue", "endpoint": self.api_queue,
             "methods": ["GET"], "summary": "获取人工队列"},
            {"path": "/confirm", "endpoint": self.api_confirm,
             "methods": ["POST"], "summary": "确认并整理"},
            {"path": "/config", "endpoint": self.api_config,
             "methods": ["POST"], "summary": "更新配置"},
            {"path": "/ai_batch", "endpoint": self.api_ai_batch,
             "methods": ["POST"], "summary": "批量AI识别"},
            {"path": "/selftest", "endpoint": self.api_selftest,
             "methods": ["POST"], "summary": "插件自检"},
            {"path": "/queue_delete", "endpoint": self.api_queue_delete,
             "methods": ["POST"], "summary": "删除队列条目"},
            {"path": "/queue_clear", "endpoint": self.api_queue_clear,
             "methods": ["POST"], "summary": "清空队列"},
            {"path": "/stats", "endpoint": self.api_stats,
             "methods": ["GET"], "summary": "获取统计"},
            {"path": "/stats_reset", "endpoint": self.api_stats_reset,
             "methods": ["POST"], "summary": "重置统计"},
        ]

    def api_queue(self, apikey: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        with self._lock:
            return {"data": list(self._queue.values())}

    def api_config(self, apikey: str = "", **kwargs):
        err = self._check_api_token(apikey)
        if err:
            return err
        changed = {}
        for k in ("ask_mode", "auto_download",
                   "threshold_auto", "threshold_manual"):
            if k in kwargs:
                val = kwargs[k]
                if k in ("threshold_auto", "threshold_manual"):
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        continue
                elif k == "auto_download":
                    val = bool(val)
                self._cfg[k] = val
                changed[k] = val
        if changed:
            self._persist()
        return {"code": 0, "msg": "ok", "changed": changed}

    def api_ai_batch(self, apikey: str = "", scope: str = "all",
                     ids: Optional[List[str]] = None):
        err = self._check_api_token(apikey)
        if err:
            return err
        cnt = 0
        with self._lock:
            if scope == "selected" and ids:
                targets = [i for i in ids if i in self._queue]
            else:
                targets = list(self._queue.keys())

        scorer = Scorer(self._cfg.get("weights") or {})
        for iid in targets:
            with self._lock:
                item = self._queue.get(iid)
                if not item:
                    continue
                item_copy = copy.deepcopy(item)
            title = item_copy.get("title") or ""
            if (item_copy.get("ai") or {}).get("name"):
                continue
            ai = self._llm_parse_cached(title)
            if ai:
                ext = item_copy.get("ext") or {}
                bd = scorer.score(ai, ext)
                with self._lock:
                    if iid in self._queue:
                        self._queue[iid]["ai"] = ai
                        self._queue[iid]["score"] = {
                            "total": bd.total, "items": bd.items}
                        cnt += 1
        if cnt > 0:
            self._persist()
        return {"code": 0, "msg": f"AI识别完成：{cnt} 条"}

    def api_confirm(self, item_id: str = "", apikey: str = "",
                    target_storage: str = "", target_path: str = "",
                    background: bool = True, id: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        _id = item_id or id
        with self._lock:
            item = self._queue.get(_id)
            if not item:
                return {"code": 404, "msg": "not found"}
            item = copy.deepcopy(item)

        body = {"items": [{
            "path": "", "type": "file",
            "target_storage": target_storage,
            "target_path": target_path,
            "mediainfo": {
                "name": ((item.get("ai") or {}).get("name")
                         or item.get("title")),
                "year": ((item.get("ai") or {}).get("year")
                         or (item.get("ext") or {}).get("year")),
                "season": _safe_int(
                    (item.get("ai") or {}).get("season")),
                "episode": _safe_int(
                    (item.get("ai") or {}).get("episode")),
                "tmdbid": (item.get("ext") or {}).get("tmdbid"),
                "doubanid": (item.get("ext") or {}).get("doubanid"),
                "bangumiid": (item.get("ext") or {}).get("bangumiid"),
                "traktid": (item.get("ext") or {}).get("traktid"),
            }}], "background": background}

        if (self._cfg.get("mp_api_base")
                and self._cfg.get("mp_api_token")):
            resp = MPClient(self._cfg["mp_api_base"],
                            self._cfg["mp_api_token"]
                            ).transfer_manual(body)
            with self._lock:
                self._queue.pop(_id, None)
            self._persist()
            return {"code": 0, "msg": "已提交整理", "resp": resp}
        else:
            with self._lock:
                self._queue.pop(_id, None)
            self._persist()
            return {"next_api": "api/v1/transfer/manual",
                    "method": "post", "json": body}

    def api_queue_delete(self, apikey: str = "", id: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        if not id:
            return {"code": 400, "msg": "missing id"}
        with self._lock:
            removed = self._queue.pop(id, None)
        if removed is None:
            return {"code": 404, "msg": "not found"}
        self._persist()
        return {"code": 0, "msg": "deleted"}

    def api_queue_clear(self, apikey: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        with self._lock:
            cnt = len(self._queue)
            self._queue.clear()
        self._persist()
        return {"code": 0, "msg": f"cleared {cnt} items"}

    def api_stats(self, apikey: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        with self._lock:
            s = {**self._stats}
        tc = s.get("total_calls", 0)
        s["rate"] = round(s.get("success", 0) / tc * 100, 1) \
            if tc > 0 else 0
        s["cache_size"] = len(self._llm_cache)
        s["queue_size"] = len(self._queue)
        return {"code": 0, "data": s}

    def api_stats_reset(self, apikey: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        with self._lock:
            self._stats = {**_STATS_DEFAULT}
        self._llm_cache.clear()
        self._persist()
        return {"code": 0, "msg": "stats and cache reset"}

    def api_selftest(self, apikey: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        logs: List[str] = []
        ok = True
        t0 = time.time()

        def log(s, good=None):
            flag = ("PASS" if good is True
                    else ("FAIL" if good is False else "INFO"))
            msg = f"[{flag}] {s}"
            logs.append(msg)
            logger.info(f"[MSAIR][SELFTEST] {msg}")

        # LLM
        llm = self._make_llm_client()
        if not llm.base or not llm.key:
            log("LLM 配置缺失（base/key），跳过。", False)
            ok = False
        else:
            try:
                resp = llm.parse_title(
                    "TEST-ONLY", "只输出JSON对象，不要解释。")
                if isinstance(resp, dict):
                    log(f"LLM OK (provider="
                        f"{self._cfg.get('llm_provider')}, "
                        f"model={llm.model}, "
                        f"timeout={llm.timeout}s)。", True)
                else:
                    log("LLM 返回非JSON或为空。", False)
                    ok = False
            except Exception as e:
                log(f"LLM 异常：{e}", False)
                ok = False

        # Telegram
        tg_token = (self._cfg.get("tg_bot_token") or "").strip()
        tg_chat = (self._cfg.get("tg_chat_id") or "").strip()
        if tg_token and tg_chat:
            try:
                r = requests.get(
                    f"https://api.telegram.org/bot{tg_token}/getMe",
                    timeout=10)
                if r.status_code == 200 and r.json().get("ok"):
                    bn = (r.json().get("result", {})
                          .get("username", "?"))
                    log(f"Telegram Bot 连通 (@{bn})。", True)
                else:
                    log(f"Telegram Token 无效: "
                        f"{r.text[:100]}", False)
                    ok = False
            except Exception as e:
                log(f"Telegram 异常: {e}", False)
                ok = False
        else:
            log("Telegram 独立通知未配置，跳过。")

        # MP API
        mp_base = self._cfg.get("mp_api_base")
        mp_tok = self._cfg.get("mp_api_token")
        if not mp_base or not mp_tok:
            log("主程序 API 未配置，跳过。")
        else:
            try:
                mp = MPClient(mp_base, mp_tok)
                r = mp.search("test", "media")
                if isinstance(r, dict):
                    log("主程序 API 可用。", True)
                else:
                    log("主程序 API 返回异常。", False)
                    ok = False
            except Exception as e:
                log(f"主程序 API 异常：{e}", False)
                ok = False

        # 打分器
        try:
            scorer = Scorer(self._cfg.get("weights") or {})
            bd = scorer.score(
                {"name": "Test", "year": "2024", "season": 1,
                 "episode": 1, "resolution": "1080p",
                 "version": None, "part": None},
                {"names": ["Test"], "year": "2024",
                 "se": {"season": 1, "episode": 1},
                 "tmdbid": 1, "is_movie": True, "imdbid": "tt1",
                 "agree_pairs": 2})
            if isinstance(bd.total, int):
                log(f"打分器正常，总分={bd.total}。", True)
            else:
                log("打分器返回异常。", False)
                ok = False
        except Exception as e:
            log(f"打分器异常：{e}", False)
            ok = False

        # 队列 dry-run
        try:
            iid = _gen_id()
            with self._lock:
                self._queue[iid] = {"id": iid, "title": "selftest"}
                ins = iid in self._queue
                del self._queue[iid]
                rem = iid not in self._queue
            if ins and rem:
                log("队列插入/删除正常。", True)
            else:
                log("队列操作异常。", False)
                ok = False
        except Exception as e:
            log(f"队列异常：{e}", False)
            ok = False

        # 统计/缓存
        log(f"缓存 {len(self._llm_cache)} 条，"
            f"统计 total={self._stats.get('total_calls', 0)}/"
            f"ok={self._stats.get('success', 0)}/"
            f"fail={self._stats.get('fail', 0)}。")

        cost = round((time.time() - t0) * 1000)
        log(f"自检完成，用时 {cost} ms。")
        self._selftest = {"ok": ok, "logs": logs,
                          "ts": int(time.time())}
        return {"code": 0 if ok else 1, "msg": "done",
                "data": self._selftest}

    # ═══════ 仪表板 ═══════

    def get_dashboard_meta(self) -> Optional[List[Dict[str, str]]]:
        return [
            {"key": "queue", "name": "AI辅助识别：待确认"},
            {"key": "stats", "name": "AI辅助识别：统计"},
        ]

    def get_dashboard(self, key: str,
                      **kwargs) -> Optional[Tuple[Dict, Dict, List]]:
        if key == "stats":
            cols = {"cols": 12, "md": 4}
            conf = {"refresh": 30, "title": "AI识别统计"}
            with self._lock:
                s = {**self._stats}
            tc = s.get("total_calls", 0)
            succ = s.get("success", 0)
            fail = s.get("fail", 0)
            rate = round(succ / tc * 100, 1) if tc > 0 else 0
            page = [{"element": "v-card",
                     "props": {"class": "pa-4"},
                     "children": [
                         {"element": "div",
                          "children": [f"总调用: {tc}"]},
                         {"element": "div",
                          "children": [f"成功: {succ}"]},
                         {"element": "div",
                          "children": [f"失败: {fail}"]},
                         {"element": "div",
                          "children": [f"成功率: {rate}%"]},
                         {"element": "div",
                          "children": [
                              f"缓存: {len(self._llm_cache)} 条"]},
                     ]}]
            return cols, conf, page

        if key == "queue":
            cols = {"cols": 12, "md": 6}
            conf = {"refresh": 10, "title": "AI待确认列表"}
            with self._lock:
                top = [
                    {"id": k, "title": v.get("title"),
                     "score": (v.get("score") or {}).get("total", 0)}
                    for k, v in list(self._queue.items())[:6]]
            page = [{"element": "v-list", "children": [
                {"element": "v-list-item",
                 "props": {"title": "{{item.title}}",
                           "subtitle": "分数：{{item.score}}"},
                 "for": {"item": top},
                 "children": [
                     {"element": "v-btn",
                      "props": {"text": True, "size": "small"},
                      "events": {"click": {
                          "api": "plugin/multisource_ai_recognizer/confirm",
                          "method": "post",
                          "json": {"id": "{{item.id}}",
                                   "target_storage": "",
                                   "target_path": ""}}},
                      "children": ["快速确认"]}]}]}]
            return cols, conf, page

        return None

    # ═══════ 工作流 ═══════

    def get_actions(self) -> List[Dict[str, Any]]:
        return [{"id": "msair_recognize", "name": "AI辅助识别评分",
                 "func": self.action_ai_recognize, "kwargs": {}}]

    def action_ai_recognize(self, action_content, **kwargs):
        title = (getattr(action_content, "title", None)
                 or getattr(action_content, "name", None))
        if not title:
            return False, action_content
        ai = self._llm_parse_cached(title)
        bd = Scorer(self._cfg.get("weights") or {}).score(ai or {}, {})
        setattr(action_content, "ext", {
            "ai": ai,
            "score": {"total": bd.total, "items": bd.items}})
        return True, action_content

    # ═══════ 消息交互 ═══════

    def get_command(self) -> List[Dict[str, Any]]:
        return [{
            "cmd": "/msair",
            "event": EventType.PluginAction,
            "desc": "AI辅助识别面板",
            "category": "插件交互",
            "data": {"action": "msair_menu"},
        }]

    @eventmanager.register(EventType.PluginAction)
    def command_action(self, event: Event):
        data = getattr(event, "event_data", {}) or {}
        if data.get("action") != "msair_menu":
            return
        channel = data.get("channel")
        user = data.get("user")
        cls_name = self.__class__.__name__
        buttons = [[
            {"text": "查看待确认",
             "callback_data": f"[PLUGIN]{cls_name}|queue"},
            {"text": "查看统计",
             "callback_data": f"[PLUGIN]{cls_name}|stats"},
            {"text": "设置",
             "callback_data": f"[PLUGIN]{cls_name}|settings"},
        ]]
        self.post_message(channel=channel, title="AI辅助识别",
                          text="请选择：", userid=user, buttons=buttons)

    @eventmanager.register(EventType.MessageAction)
    def message_action(self, event: Event):
        data = getattr(event, "event_data", {}) or {}
        if data.get("plugin_id") != self.__class__.__name__:
            return
        text = data.get("text", "")
        channel = data.get("channel")
        user = data.get("userid")
        if text == "queue":
            with self._lock:
                cnt = len(self._queue)
            self.post_message(channel=channel, title="队列",
                              text=f"当前待确认 {cnt} 条", userid=user)
        elif text == "stats":
            with self._lock:
                s = {**self._stats}
            tc = s.get("total_calls", 0)
            succ = s.get("success", 0)
            fail = s.get("fail", 0)
            rate = round(succ / tc * 100, 1) if tc > 0 else 0
            self.post_message(
                channel=channel, title="识别统计",
                text=(f"总调用: {tc}\n成功: {succ}\n"
                      f"失败: {fail}\n成功率: {rate}%"),
                userid=user)
        elif text == "settings":
            self.post_message(channel=channel, title="设置",
                              text="请在插件页面中修改配置。",
                              userid=user)
