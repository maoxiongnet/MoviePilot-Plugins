# -*- coding: utf-8 -*-
"""
AI 辅助识别插件 for MoviePilot
- 使用 LLM (JSON-only) 辅助解析媒体标题
- 积分制评分（可>100），阈值可配
- 智能触发AI：仅当主程序识别结果不完整时才问AI；也支持"总是/手动"
- 自动下载（>=阈值可选），人工队列手动确认并可选择目录（transfer/manual）
- 仪表板、按钮交互、工作流、消息回调、配置页
- 自检按钮：LLM/主程序API/打分器/队列流程健康检查 + 详细日志
"""

from __future__ import annotations

import copy
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
# === MoviePilot 插件基类兼容层 ===
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
                "无法从 app.core.plugin 导入 PluginBase/Plugin —— 当前 MoviePilot 版本或分支的插件基类命名不一致。"
                "请升级 MoviePilot，或保留此兼容层。"
            ) from e
from app.log import logger
from app.core.config import settings


# ========= LLM System Prompt（全局常量，避免重复）=========
LLM_SYSTEM_PROMPT = (
    "你是严格模式的命名解析器。只输出一个JSON对象，不得包含解释、Markdown或代码块。"
    "字段：name, version, part, year, resolution, season, episode。"
    "规则：year为4位数字或null；season/episode为正整数或null；其余为字符串或null。"
    '无法解析时输出{}。示例：'
    '{"name":"xxx","year":"2024","season":1,"episode":2,"version":null,"part":null,"resolution":"1080p"}'
)


# ========= 默认权重与阈值 =========
SCORE_WEIGHTS_DEFAULT = {
    # ID命中
    "tmdb_hit": 60,
    "douban_hit": 50,
    "bangumi_hit": 55,
    "trakt_hit": 40,
    # 结构匹配
    "title_sim_max": 30,
    "year_match_max": 15,
    "se_match_max": 25,
    # 多源一致性（两两一致每对+12，上限+36）
    "consistency_pair": 12,
    # 领域加成
    "anime_bangumi_bonus": 10,
    "movie_imdb_bonus": 10,
    # AI贡献
    "ai_structured": 15,
    "ai_field_completeness_max": 15,
    # 惩罚项
    "year_conflict": -30,
    "id_conflict": -50,
    "unstructured_penalty": -20,
}
THRESHOLD_AUTO_DEFAULT = 120
THRESHOLD_MANUAL_DEFAULT = 80


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
    """
    标题相似度（0~1），使用 SequenceMatcher 进行序列级比较。
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ========= LLM 调用（仅 JSON 模式）=========
class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 12):
        self.base = (base_url or "").rstrip("/")
        self.key = api_key or ""
        self.model = model or "deepseek-v3"
        self.timeout = timeout

    def parse_title(self, title: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """
        只返回 dict；失败返回 None
        """
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
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return self._robust_json(content)
        except Exception as e:
            logger.warning(f"[MSAIR][LLM] request failed: {e}")
            return None

    @staticmethod
    def _robust_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                pass
        try:
            return json.loads(text)
        except Exception:
            return None


# ========= MP 后端直连（可选）=========
class MPClient:
    def __init__(self, base: str = "", token: str = "", timeout: int = 6):
        self.base = base.rstrip("/") if base else ""
        self.token = token or ""
        self.timeout = timeout

    def _headers(self):
        return (
            {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            if self.token
            else {"Content-Type": "application/json"}
        )

    def search(self, title: str, mtype: str = "media") -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.get(
                f"{self.base}/api/v1/media/search",
                params={"title": title, "type": mtype},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug(f"[MSAIR][MP] search error: {e}")
        return None

    def transfer_manual(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.post(
                f"{self.base}/api/v1/transfer/manual",
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
            )
            return r.json()
        except Exception as e:
            logger.debug(f"[MSAIR][MP] transfer_manual error: {e}")
            return None

    def download_with_media(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.post(
                f"{self.base}/api/v1/download/",
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
            )
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

    def score(self, ai: Dict[str, Any], ext: Dict[str, Any]) -> ScoreBreakdown:
        bd = ScoreBreakdown()
        name = (ai or {}).get("name") or ""
        year = (ai or {}).get("year")
        season = _safe_int((ai or {}).get("season"))
        episode = _safe_int((ai or {}).get("episode"))

        # AI 贡献
        if ai and isinstance(ai, dict) and ai.get("name"):
            bd.add("ai_structured", self.w["ai_structured"])
            fields = ["name", "year", "season", "episode", "resolution", "version", "part"]
            filled = sum(1 for f in fields if ai.get(f) not in (None, "", []))
            bd.add(
                "ai_field_completeness",
                int(round(self.w["ai_field_completeness_max"] * (filled / len(fields)))),
            )
        elif ai is not None:
            bd.add("unstructured_penalty", self.w["unstructured_penalty"])

        # ID 命中
        if ext.get("tmdbid"):
            bd.add("tmdb_hit", self.w["tmdb_hit"])
        if ext.get("doubanid"):
            bd.add("douban_hit", self.w["douban_hit"])
        if ext.get("bangumiid"):
            bd.add("bangumi_hit", self.w["bangumi_hit"])
        if ext.get("traktid"):
            bd.add("trakt_hit", self.w["trakt_hit"])

        # ID 冲突检测：如果 ext 中标记了来自不同源的 ID 指向不同媒体
        if ext.get("id_conflict"):
            bd.add("id_conflict", self.w["id_conflict"])

        # 标题相似度
        names = ext.get("names") or []
        if name and names:
            sims = [_title_similarity(name, n) for n in names]
            bd.add("title_sim", int(round(max(sims) * self.w["title_sim_max"])))

        # 年份匹配
        y_mp = ext.get("year")
        if year and y_mp:
            try:
                if str(year) == str(y_mp):
                    bd.add("year_match", self.w["year_match_max"])
                elif abs(int(year) - int(y_mp)) == 1:
                    bd.add("year_match_near", int(round(self.w["year_match_max"] * 0.6)))
                else:
                    bd.add("year_conflict", self.w["year_conflict"])
            except Exception:
                pass

        # 季/集匹配
        se_mp = ext.get("se") or {}
        if season is not None and se_mp.get("season") == season:
            bd.add("season_match", int(round(self.w["se_match_max"] * 0.6)))
        if episode is not None and se_mp.get("episode") == episode:
            bd.add("episode_match", int(round(self.w["se_match_max"] * 0.4)))

        # 多源一致性（由 ext["agree_pairs"] 提供数量，上限3对）
        if _safe_int(ext.get("agree_pairs")):
            bd.add("consistency_bonus", self.w["consistency_pair"] * min(3, int(ext["agree_pairs"])))

        # 领域加成
        if ext.get("is_anime") and ext.get("bangumiid"):
            bd.add("anime_bangumi_bonus", self.w["anime_bangumi_bonus"])
        if ext.get("is_movie") and ext.get("imdbid") and ext.get("tmdbid"):
            bd.add("movie_imdb_bonus", self.w["movie_imdb_bonus"])

        return bd


# ========= 插件主体 =========
class Multisource_Ai_Recognizer(MPPluginBase):
    """
    AI 辅助识别与评分
    - 类名使用带下划线驼峰，便于与目录名/清单键名对齐（目录小写：multisource_ai_recognizer）
    """

    plugin_name = "AI辅助识别与评分"
    plugin_desc = (
        "LLM 辅助媒体标题解析；积分制评分（可>100）；低分入人工队列并支持自选目录/自动下载"
    )
    plugin_icon = "https://raw.githubusercontent.com/jxxghp/MoviePilot-Plugins/main/icons/chatgpt.png"
    plugin_version = "1.5.0"
    plugin_author = "maoxiongnet"
    plugin_order = 50

    def __init__(self):
        super().__init__()
        self._cfg: Dict[str, Any] = {}
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._selftest: Dict[str, Any] = {}

    def init_plugin(self, config: dict = None):
        """
        插件初始化（MoviePilot 标准生命周期方法）
        """
        default_cfg = {
            "llm_base": "https://api.gptapi.us/v1",
            "llm_model": "deepseek-v3",
            "llm_key": "",
            "mp_api_base": "",
            "mp_api_token": "",
            "ask_mode": "smart",
            "auto_download": False,
            "threshold_auto": THRESHOLD_AUTO_DEFAULT,
            "threshold_manual": THRESHOLD_MANUAL_DEFAULT,
            "weights": SCORE_WEIGHTS_DEFAULT,
        }
        if config:
            default_cfg.update(config)
            # 恢复持久化的队列
            saved_queue = config.get("_queue")
            if isinstance(saved_queue, dict):
                with self._lock:
                    self._queue = saved_queue
        self._cfg = default_cfg

    def _make_llm_client(self) -> LLMClient:
        return LLMClient(
            self._cfg.get("llm_base", ""),
            self._cfg.get("llm_key", ""),
            self._cfg.get("llm_model", ""),
        )

    def _save_queue(self):
        """将队列持久化到插件配置中"""
        try:
            with self._lock:
                queue_snapshot = copy.deepcopy(self._queue)
            self.update_config({**self._cfg, "_queue": queue_snapshot})
        except Exception as e:
            logger.debug(f"[MSAIR] save queue failed: {e}")

    # ===== 配置页（可视化） =====
    def get_setting(self) -> Optional[dict]:
        return {
            "element": "v-container",
            "props": {"fluid": True},
            "children": [
                {
                    "element": "v-row",
                    "children": [
                        {
                            "element": "v-col",
                            "props": {"cols": 12, "md": 6},
                            "children": [
                                {
                                    "element": "v-text-field",
                                    "props": {
                                        "label": "LLM Base URL",
                                        "model": "llm_base",
                                        "placeholder": "https://api.gptapi.us/v1",
                                    },
                                },
                                {
                                    "element": "v-text-field",
                                    "props": {"label": "LLM Model", "model": "llm_model", "placeholder": "deepseek-v3"},
                                },
                                {
                                    "element": "v-text-field",
                                    "props": {"label": "LLM API Key", "model": "llm_key", "type": "password"},
                                },
                                {
                                    "element": "v-text-field",
                                    "props": {
                                        "label": "MoviePilot API 地址（可选）",
                                        "model": "mp_api_base",
                                        "placeholder": "http://localhost:3000",
                                    },
                                },
                                {
                                    "element": "v-text-field",
                                    "props": {
                                        "label": "MoviePilot API Token（可选）",
                                        "model": "mp_api_token",
                                        "type": "password",
                                    },
                                },
                            ],
                        },
                        {
                            "element": "v-col",
                            "props": {"cols": 12, "md": 6},
                            "children": [
                                {
                                    "element": "v-select",
                                    "props": {
                                        "label": "AI触发模式",
                                        "model": "ask_mode",
                                        "items": [
                                            {"title": "智能(smart)", "value": "smart"},
                                            {"title": "总是(always)", "value": "always"},
                                            {"title": "手动(manual)", "value": "manual"},
                                        ],
                                    },
                                },
                                {"element": "v-switch", "props": {"label": "自动下载（>=自动阈值）", "model": "auto_download"}},
                                {"element": "v-text-field", "props": {"label": "自动通过阈值", "model": "threshold_auto", "type": "number"}},
                                {"element": "v-text-field", "props": {"label": "人工队列阈值（下限）", "model": "threshold_manual", "type": "number"}},
                            ],
                        },
                    ],
                },
                {
                    "element": "v-alert",
                    "props": {"type": "info", "text": True},
                    "children": [
                        "打分：AI贡献/标题相似度/年份/季集匹配；分数可 >100。推荐阈值：自动>=120；人工80-119；<80交由主程序处理。"
                    ],
                },
            ],
        }

    def get_state(self) -> Optional[dict]:
        return {**self._cfg, "_queue": self._queue}

    def set_state(self, state: dict):
        if not state:
            return
        saved_queue = state.get("_queue")
        cfg = {k: v for k, v in state.items() if k != "_queue"}
        self._cfg.update(cfg)
        if isinstance(saved_queue, dict):
            with self._lock:
                self._queue = saved_queue

    # ===== NameRecognize 链式事件处理 =====
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

        # manual 模式下不自动触发
        if mode == "manual":
            return

        # smart 模式：检查 event_data 中是否已有可靠的识别结果
        # 如果主程序或其它插件已经填充了 name 字段，说明识别成功，无需 AI 介入
        if mode == "smart":
            existing_name = getattr(data, "name", None)
            if existing_name:
                logger.debug(f"[MSAIR] smart mode: already recognized as '{existing_name}', skipping AI")
                return

        # 调 AI 解析标题
        llm = self._make_llm_client()
        ai = llm.parse_title(title, LLM_SYSTEM_PROMPT)
        if not ai or not ai.get("name"):
            logger.debug(f"[MSAIR] LLM returned empty result for: {title}")
            return

        # 将 AI 识别结果写回 event_data，使主程序能接收到
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

        logger.info(f"[MSAIR] AI recognized: name={ai_name}, year={ai_year}, "
                     f"S{ai_season}E{ai_episode} for title: {title}")

        # 评分（用于队列管理，不影响主程序识别流程）
        # 从 event_data 中提取已有识别信息，供评分器使用
        ext: Dict[str, Any] = {}
        # ID 命中
        for attr in ("tmdbid", "doubanid", "bangumiid", "traktid", "imdbid"):
            val = getattr(data, attr, None)
            if val:
                ext[attr] = val
        # 标题候选列表（用于相似度比较）
        existing_name = getattr(data, "name", None)
        if existing_name:
            ext["names"] = [existing_name]
        # 年份
        existing_year = getattr(data, "year", None)
        if existing_year:
            ext["year"] = str(existing_year)
        # 季/集
        existing_season = _safe_int(getattr(data, "season", None))
        existing_episode = _safe_int(getattr(data, "episode", None))
        if existing_season is not None or existing_episode is not None:
            ext["se"] = {}
            if existing_season is not None:
                ext["se"]["season"] = existing_season
            if existing_episode is not None:
                ext["se"]["episode"] = existing_episode
        # 类型标记
        mtype = getattr(data, "type", None) or getattr(data, "media_type", None)
        if mtype:
            mtype_str = str(mtype).lower()
            if "anime" in mtype_str or "动漫" in mtype_str:
                ext["is_anime"] = True
            if "movie" in mtype_str or "电影" in mtype_str:
                ext["is_movie"] = True

        scorer = Scorer(self._cfg.get("weights") or {})
        bd = scorer.score(ai, ext)
        total = bd.total
        th_auto = int(self._cfg.get("threshold_auto", THRESHOLD_AUTO_DEFAULT))
        th_manual = int(self._cfg.get("threshold_manual", THRESHOLD_MANUAL_DEFAULT))

        # 自动下载（可选）
        if total >= th_auto and self._cfg.get("auto_download"):
            body = {
                "mediainfo": {
                    "title": ai_name or title,
                    "year": ai_year,
                    "season": ai_season,
                    "episode": ai_episode,
                }
            }
            if self._cfg.get("mp_api_base") and self._cfg.get("mp_api_token"):
                resp = MPClient(self._cfg["mp_api_base"], self._cfg["mp_api_token"]).download_with_media(body)
                logger.info(f"[MSAIR] auto download resp: {resp}")
            else:
                iid = _gen_id()
                with self._lock:
                    self._queue[iid] = {
                        "id": iid,
                        "title": title,
                        "ai": ai,
                        "ext": ext,
                        "score": {"total": total, "items": bd.items},
                        "auto_download_payload": body,
                    }
                self._save_queue()
            return

        # 入人工队列
        if total >= th_manual:
            iid = _gen_id()
            with self._lock:
                self._queue[iid] = {
                    "id": iid,
                    "title": title,
                    "ai": ai,
                    "ext": ext,
                    "score": {"total": total, "items": bd.items},
                }
            self._save_queue()

    # ===== 页面：人工队列 + 批量AI + 自检 =====
    def get_page(self) -> Optional[dict]:
        rows = []
        with self._lock:
            for k, v in self._queue.items():
                rows.append(
                    {
                        "id": k,
                        "title": v.get("title"),
                        "name": (v.get("ai") or {}).get("name"),
                        "year": (v.get("ai") or {}).get("year"),
                        "score": (v.get("score") or {}).get("total", 0),
                    }
                )

        top_controls = {
            "element": "v-row",
            "children": [
                {
                    "element": "v-col",
                    "props": {"cols": 12, "md": 6},
                    "children": [
                        {
                            "element": "v-btn",
                            "props": {"color": "primary", "class": "mr-2"},
                            "events": {
                                "click": {
                                    "api": "plugin/multisource_ai_recognizer/ai_batch",
                                    "method": "post",
                                    "json": {"scope": "all"},
                                }
                            },
                            "children": ["AI识别（全部）"],
                        },
                        {
                            "element": "v-btn",
                            "props": {"color": "primary", "class": "mr-2"},
                            "events": {
                                "click": {
                                    "api": "plugin/multisource_ai_recognizer/ai_batch",
                                    "method": "post",
                                    "json": {"scope": "selected", "ids": "{{selectedIds}}"},
                                }
                            },
                            "children": ["AI识别（所选）"],
                        },
                        {
                            "element": "v-btn",
                            "props": {"color": "secondary"},
                            "events": {"click": {"api": "plugin/multisource_ai_recognizer/selftest", "method": "post"}},
                            "children": ["自检"],
                        },
                    ],
                },
                {
                    "element": "v-col",
                    "props": {"cols": 12, "md": 6},
                    "children": [
                        {
                            "element": "v-select",
                            "props": {
                                "label": "AI触发模式",
                                "model": "ask_mode",
                                "items": [
                                    {"title": "智能(smart)", "value": "smart"},
                                    {"title": "总是(always)", "value": "always"},
                                    {"title": "手动(manual)", "value": "manual"},
                                ],
                                "value": self._cfg.get("ask_mode", "smart"),
                            },
                            "events": {
                                "change": {
                                    "api": "plugin/multisource_ai_recognizer/config",
                                    "method": "post",
                                    "json": {"ask_mode": "{{ask_mode}}"},
                                }
                            },
                        }
                    ],
                },
            ],
        }

        target_inputs = {
            "element": "v-row",
            "children": [
                {
                    "element": "v-col",
                    "props": {"cols": 12, "md": 4},
                    "children": [
                        {
                            "element": "v-text-field",
                            "props": {
                                "label": "目标存储类型",
                                "model": "target_storage",
                                "placeholder": "local/nas/...",
                            },
                        }
                    ],
                },
                {
                    "element": "v-col",
                    "props": {"cols": 12, "md": 8},
                    "children": [
                        {
                            "element": "v-text-field",
                            "props": {"label": "目标目录路径", "model": "target_path", "placeholder": "/media/Movies"},
                        }
                    ],
                },
            ],
        }

        table = {
            "element": "v-data-table",
            "props": {
                "items": rows,
                "show-select": True,
                "model": "selectedIds",
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
                        "props": {"text": True, "color": "primary"},
                        "events": {
                            "click": {
                                "api": "plugin/multisource_ai_recognizer/confirm",
                                "method": "post",
                                "json": {
                                    "id": "{{item.id}}",
                                    "target_storage": "{{target_storage}}",
                                    "target_path": "{{target_path}}",
                                },
                            }
                        },
                        "children": ["确认并整理"],
                    }
                },
            },
        }

        children = [top_controls, target_inputs, table]

        if self._selftest:
            ok_flag = bool(self._selftest.get("ok"))
            logs_list = self._selftest.get("logs") or []
            txt = "\n".join(logs_list) if logs_list else "（无日志）"
            result_text = "自检结果：" + ("通过" if ok_flag else "存在问题")

            children.append(
                {
                    "element": "v-alert",
                    "props": {"type": "success" if ok_flag else "error", "text": True},
                    "children": [result_text],
                }
            )
            children.append(
                {
                    "element": "v-card",
                    "children": [
                        {"element": "v-card-title", "children": ["自检日志"]},
                        {"element": "v-card-text", "children": [txt]},
                    ],
                }
            )

        return {"element": "v-container", "props": {"fluid": True}, "children": children}

    @staticmethod
    def _check_api_token(apikey: str = "") -> Optional[dict]:
        """校验 API Token，返回 None 表示通过，否则返回错误响应"""
        if apikey != settings.API_TOKEN:
            return {"code": 403, "msg": "API Token 验证失败"}
        return None

    # ===== 页面 API =====
    def get_api(self) -> Optional[List[dict]]:
        return [
            {"path": "/queue", "endpoint": self.api_queue, "methods": ["GET"], "summary": "获取人工队列"},
            {"path": "/confirm", "endpoint": self.api_confirm, "methods": ["POST"], "summary": "确认并整理"},
            {"path": "/config", "endpoint": self.api_config, "methods": ["POST"], "summary": "更新配置（ask_mode/阈值）"},
            {"path": "/ai_batch", "endpoint": self.api_ai_batch, "methods": ["POST"], "summary": "批量AI识别（全部/所选）"},
            {"path": "/selftest", "endpoint": self.api_selftest, "methods": ["POST"], "summary": "插件自检（LLM/评分/流程）"},
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
        for k in ("ask_mode", "auto_download", "threshold_auto", "threshold_manual"):
            if k in kwargs:
                val = kwargs[k]
                # 类型校验与转换
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
            self._save_queue()
        return {"code": 0, "msg": "ok", "changed": changed}

    def api_ai_batch(self, apikey: str = "", scope: str = "all", ids: Optional[List[str]] = None):
        err = self._check_api_token(apikey)
        if err:
            return err
        cnt = 0
        targets: List[str] = []
        with self._lock:
            if scope == "selected" and ids:
                targets = [i for i in ids if i in self._queue]
            else:
                targets = list(self._queue.keys())

        llm = self._make_llm_client()
        scorer = Scorer(self._cfg.get("weights") or {})

        for iid in targets:
            with self._lock:
                item = self._queue.get(iid)
                if not item:
                    continue
                # Deep copy to avoid race conditions
                item_copy = copy.deepcopy(item)

            title = item_copy.get("title") or ""
            if (item_copy.get("ai") or {}).get("name"):
                continue

            ai = llm.parse_title(title, LLM_SYSTEM_PROMPT)
            if ai:
                # Re-score with new AI result
                ext = item_copy.get("ext") or {}
                bd = scorer.score(ai, ext)
                with self._lock:
                    if iid in self._queue:
                        self._queue[iid]["ai"] = ai
                        self._queue[iid]["score"] = {"total": bd.total, "items": bd.items}
                        cnt += 1

        if cnt > 0:
            self._save_queue()
        return {"code": 0, "msg": f"AI识别完成：{cnt} 条"}

    def api_confirm(self, item_id: str = "", apikey: str = "", target_storage: str = "", target_path: str = "", background: bool = True,
                    id: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        # 兼容旧参数名 id，优先使用 item_id
        _id = item_id or id
        with self._lock:
            item = self._queue.get(_id)
            if not item:
                return {"code": 404, "msg": "not found"}
            # Deep copy under lock to avoid race conditions
            item = copy.deepcopy(item)

        body = {
            "items": [
                {
                    "path": "",
                    "type": "file",
                    "target_storage": target_storage,
                    "target_path": target_path,
                    "mediainfo": {
                        "name": (item.get("ai") or {}).get("name") or item.get("title"),
                        "year": (item.get("ai") or {}).get("year") or (item.get("ext") or {}).get("year"),
                        "season": _safe_int((item.get("ai") or {}).get("season")),
                        "episode": _safe_int((item.get("ai") or {}).get("episode")),
                        "tmdbid": (item.get("ext") or {}).get("tmdbid"),
                        "doubanid": (item.get("ext") or {}).get("doubanid"),
                        "bangumiid": (item.get("ext") or {}).get("bangumiid"),
                        "traktid": (item.get("ext") or {}).get("traktid"),
                    },
                }
            ],
            "background": background,
        }

        if self._cfg.get("mp_api_base") and self._cfg.get("mp_api_token"):
            resp = MPClient(self._cfg["mp_api_base"], self._cfg["mp_api_token"]).transfer_manual(body)
            with self._lock:
                self._queue.pop(_id, None)
            self._save_queue()
            return {"code": 0, "msg": "已提交整理", "resp": resp}
        else:
            with self._lock:
                self._queue.pop(_id, None)
            self._save_queue()
            return {"next_api": "api/v1/transfer/manual", "method": "post", "json": body}

    def api_selftest(self, apikey: str = ""):
        err = self._check_api_token(apikey)
        if err:
            return err
        logs: List[str] = []
        ok = True
        t0 = time.time()

        def log(s, good=None):
            flag = "PASS" if good is True else ("FAIL" if good is False else "INFO")
            msg = f"[{flag}] {s}"
            logs.append(msg)
            logger.info(f"[MSAIR][SELFTEST] {msg}")

        # 1) LLM 测试
        base, key, model = self._cfg.get("llm_base"), self._cfg.get("llm_key"), self._cfg.get("llm_model")
        if not base or not key:
            log("LLM 配置缺失（llm_base/llm_key），跳过 LLM 自测。", False)
            ok = False
        else:
            try:
                llm = LLMClient(base, key, model)
                sys_prompt = "只输出JSON对象，不要解释。"
                resp = llm.parse_title("TEST-ONLY", sys_prompt)
                if isinstance(resp, dict):
                    log("LLM JSON 模式兼容（/chat/completions + response_format）。", True)
                else:
                    log("LLM 返回非JSON或为空。", False)
                    ok = False
            except Exception as e:
                log(f"LLM 测试异常：{e}", False)
                ok = False

        # 2) 主程序 API 测试（可选）
        mp_base = self._cfg.get("mp_api_base")
        mp_tok = self._cfg.get("mp_api_token")
        if not mp_base or not mp_tok:
            log("主程序后端直连未配置（mp_api_base/mp_api_token），跳过直连自测。")
        else:
            try:
                mp = MPClient(mp_base, mp_tok)
                r = mp.search("test", "media")
                if isinstance(r, dict):
                    log("主程序 API 直连可用（/api/v1/media/search）。", True)
                else:
                    log("主程序 API 直连返回异常。", False)
                    ok = False
            except Exception as e:
                log(f"主程序 API 直连异常：{e}", False)
                ok = False

        # 3) 打分器健壮性
        try:
            scorer = Scorer(self._cfg.get("weights") or {})
            ai = {
                "name": "Demo Title",
                "year": "2024",
                "season": 1,
                "episode": 1,
                "resolution": "1080p",
                "version": None,
                "part": None,
            }
            ext = {
                "names": ["Demo Title", "演示标题"],
                "year": "2024",
                "se": {"season": 1, "episode": 1},
                "tmdbid": 1,
                "is_movie": True,
                "imdbid": "tt123",
                "agree_pairs": 2,
            }
            bd = scorer.score(ai, ext)
            if isinstance(bd.total, int):
                log(f"打分器运行正常，总分={bd.total}。", True)
            else:
                log("打分器返回异常类型。", False)
                ok = False
        except Exception as e:
            log(f"打分器异常：{e}", False)
            ok = False

        # 4) 队列与确认流程（dry-run，不发送真实请求）
        try:
            iid = _gen_id()
            test_item = {
                "id": iid,
                "title": "SelfTest Item 2024 S01E01 1080p",
                "ai": {"name": "SelfTest Item", "year": "2024", "season": 1, "episode": 1},
                "ext": {"tmdbid": 1},
                "score": {"total": 130, "items": [("demo", 130)]},
            }
            # 验证队列插入和删除逻辑，但不调用 api_confirm 以避免真实 API 请求
            with self._lock:
                self._queue[iid] = test_item
                inserted = iid in self._queue
                del self._queue[iid]
                removed = iid not in self._queue
            if inserted and removed:
                log("队列插入/删除流程正常。", True)
            else:
                log("队列插入/删除流程异常。", False)
                ok = False
            # 验证 confirm payload 构造逻辑
            item = test_item
            body = {
                "items": [{
                    "path": "",
                    "type": "file",
                    "target_storage": "local",
                    "target_path": "/media/SelfTest",
                    "mediainfo": {
                        "name": (item.get("ai") or {}).get("name") or item.get("title"),
                        "year": (item.get("ai") or {}).get("year"),
                        "season": _safe_int((item.get("ai") or {}).get("season")),
                        "episode": _safe_int((item.get("ai") or {}).get("episode")),
                    },
                }],
                "background": True,
            }
            if isinstance(body, dict) and body.get("items"):
                log("确认 payload 构造正常（dry-run，未发送真实请求）。", True)
            else:
                log("确认 payload 构造异常。", False)
                ok = False
        except Exception as e:
            log(f"确认流程异常：{e}", False)
            ok = False

        cost = round((time.time() - t0) * 1000)
        log(f"自检完成，用时 {cost} ms。")
        self._selftest = {"ok": ok, "logs": logs, "ts": int(time.time())}
        return {"code": 0 if ok else 1, "msg": "done", "data": self._selftest}

    # ===== 仪表板 =====
    def get_dashboard_meta(self) -> Optional[List[Dict[str, str]]]:
        return [{"key": "queue", "name": "AI辅助识别：待确认"}]

    def get_dashboard(self, key: str, **kwargs) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], List[dict]]]:
        if key != "queue":
            return None
        cols = {"cols": 12, "md": 6}
        conf = {"refresh": 10, "title": "AI待确认列表"}
        with self._lock:
            top = [
                {"id": k, "title": v.get("title"), "score": (v.get("score") or {}).get("total", 0)}
                for k, v in list(self._queue.items())[:6]
            ]
        page = [
            {
                "element": "v-list",
                "children": [
                    {
                        "element": "v-list-item",
                        "props": {"title": "{{item.title}}", "subtitle": "分数：{{item.score}}"},
                        "for": {"item": top},
                        "children": [
                            {
                                "element": "v-btn",
                                "props": {"text": True, "size": "small"},
                                "events": {
                                    "click": {
                                        "api": "plugin/multisource_ai_recognizer/confirm",
                                        "method": "post",
                                        "json": {
                                            "id": "{{item.id}}",
                                            "target_storage": "",
                                            "target_path": "",
                                        },
                                    }
                                },
                                "children": ["快速确认"],
                            }
                        ],
                    }
                ],
            }
        ]
        return cols, conf, page

    # ===== 工作流（v2.4.8+） =====
    def get_actions(self) -> List[Dict[str, Any]]:
        return [{"id": "msair_recognize", "name": "AI辅助识别评分", "func": self.action_ai_recognize, "kwargs": {}}]

    def action_ai_recognize(self, action_content, **kwargs):
        title = getattr(action_content, "title", None) or getattr(action_content, "name", None)
        if not title:
            return False, action_content
        llm = self._make_llm_client()
        ai = llm.parse_title(title, LLM_SYSTEM_PROMPT)
        bd = Scorer(self._cfg.get("weights") or {}).score(ai or {}, {})
        setattr(action_content, "ext", {"ai": ai, "score": {"total": bd.total, "items": bd.items}})
        return True, action_content

    # ===== 消息交互（v2.5.7+） =====
    def get_command(self) -> List[Dict[str, Any]]:
        return [
            {
                "cmd": "/msair",
                "event": EventType.PluginAction,
                "desc": "AI辅助识别面板",
                "category": "插件交互",
                "data": {"action": "msair_menu"},
            }
        ]

    @eventmanager.register(EventType.PluginAction)
    def command_action(self, event: Event):
        data = getattr(event, "event_data", {}) or {}
        if data.get("action") != "msair_menu":
            return
        channel = data.get("channel")
        user = data.get("user")
        buttons = [
            [
                {"text": "查看待确认", "callback_data": f"[PLUGIN]{self.__class__.__name__}|queue"},
                {"text": "设置", "callback_data": f"[PLUGIN]{self.__class__.__name__}|settings"},
            ]
        ]
        self.post_message(channel=channel, title="AI辅助识别", text="请选择：", userid=user, buttons=buttons)

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
            self.post_message(channel=channel, title="队列", text=f"当前待确认 {cnt} 条", userid=user)
        elif text == "settings":
            self.post_message(channel=channel, title="设置", text="请在插件页面中修改配置。", userid=user)
