# -*- coding: utf-8 -*-
"""
Multisource AI Recognizer for MoviePilot
- LLM(JSON-only) + Douban/Trakt/Bangumi/TMDB 多源互证
- 积分制评分（可>100），阈值可配
- 智能触发AI：仅当不确定/模糊时才问AI；也支持“全部问/只问所选”
- 自动下载（≥阈值可选），人工队列手动确认并可选择目录（transfer/manual）
- 仪表板、按钮交互、工作流、消息回调、配置页
- 🧪 自检按钮：LLM/主程序API/打分器/队列流程健康检查 + 详细日志
"""

from __future__ import annotations
import json
import re
import time
import threading
import random
import string
from typing import Any, Dict, List, Optional, Tuple

import requests

# ===== MoviePilot 基础导入 =====
from app.core.event import eventmanager, Event, EventType, ChainEventType
from app.core.plugin import PluginBase
from app.log import logger

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
        if x is None: return default
        return int(x)
    except Exception:
        return default

def _gen_id(n=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))

def _title_similarity(a: str, b: str) -> float:
    """
    非严格相似度（0~1），简化实现：字符集 Jaccard
    """
    if not a or not b:
        return 0.0
    sa, sb = set(a.lower()), set(b.lower())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

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
                {"role": "user", "content": f"解析以下标题为JSON：{title}"}
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
                return json.loads(text[i:j+1])
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
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"} if self.token else {"Content-Type": "application/json"}

    def recognize(self, title: str, subtitle: str = "") -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.get(f"{self.base}/api/v1/media/recognize",
                             params={"title": title, "subtitle": subtitle},
                             headers=self._headers(), timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug(f"[MSAIR][MP] recognize error: {e}")
        return None

    def search(self, title: str, mtype: str = "media") -> Optional[Dict[str, Any]]:
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

    def transfer_manual(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.post(f"{self.base}/api/v1/transfer/manual",
                              headers=self._headers(), json=body, timeout=self.timeout)
            return r.json()
        except Exception as e:
            logger.debug(f"[MSAIR][MP] transfer_manual error: {e}")
            return None

    def download_with_media(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.base:
            return None
        try:
            r = requests.post(f"{self.base}/api/v1/download/",
                              headers=self._headers(), json=body, timeout=self.timeout)
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
    def score(self, title: str, ai: Dict[str, Any], ext: Dict[str, Any]) -> ScoreBreakdown:
        bd = ScoreBreakdown()
        name = (ai or {}).get("name") or ""
        year = (ai or {}).get("year")
        season = _safe_int((ai or {}).get("season"))
        episode = _safe_int((ai or {}).get("episode"))

        # AI 贡献
        if ai and isinstance(ai, dict):
            bd.add("ai_structured", self.w["ai_structured"])
            fields = ["name", "year", "season", "episode", "resolution", "version", "part"]
            filled = sum(1 for f in fields if ai.get(f) not in (None, "", []))
            bd.add("ai_field_completeness", int(round(self.w["ai_field_completeness_max"] * (filled / len(fields)))))
        else:
            bd.add("unstructured_penalty", self.w["unstructured_penalty"])

        # ID 命中
        if ext.get("tmdbid"): bd.add("tmdb_hit", self.w["tmdb_hit"])
        if ext.get("doubanid"): bd.add("douban_hit", self.w["douban_hit"])
        if ext.get("bangumiid"): bd.add("bangumi_hit", self.w["bangumi_hit"])
        if ext.get("traktid"): bd.add("trakt_hit", self.w["trakt_hit"])

        # 标题相似度
        names = ext.get("names") or []
        if name and names:
            sims = [ _title_similarity(name, n) for n in names ]
            bd.add("title_sim", int(round(max(sims) * self.w["title_sim_max"])) )

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
        if season and se_mp.get("season") == season:
            bd.add("season_match", int(round(self.w["se_match_max"] * 0.6)))
        if episode and se_mp.get("episode") == episode:
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
class MultisourceAIRecognizer(PluginBase):
    plugin_name = "多源AI识别与评分"
    plugin_desc = "LLM + Douban/Trakt/Bangumi/TMDB 多源互证；积分制（可>100）；低分入人工队列并支持自选目录/自动下载"
    plugin_icon = "https://raw.githubusercontent.com/jxxghp/MoviePilot-Plugins/main/icons/chatgpt.png"
    plugin_version = "1.3.0"

    def __init__(self):
        super().__init__()
        self._cfg = {
            # LLM
            "llm_base": "https://api.gptapi.us/v1",
            "llm_model": "deepseek-v3",
            "llm_key": "",
            # MP 后端直连（可留空）
            "mp_api_base": "",
            "mp_api_token": "",
            # 行为
            "ask_mode": "smart",      # smart/always/manual
            "auto_download": False,   # ≥阈值时自动下载
            "threshold_auto": THRESHOLD_AUTO_DEFAULT,
            "threshold_manual": THRESHOLD_MANUAL_DEFAULT,
            "weights": SCORE_WEIGHTS_DEFAULT,
        }
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._selftest: Dict[str, Any] = {}

    # ===== 配置页（可视化） =====
    def get_setting(self) -> Optional[dict]:
        return {
            "element": "v-container",
            "props": {"fluid": True},
            "children": [
                {"element": "v-row", "children": [
                    {"element": "v-col", "props": {"cols": 12, "md": 6}, "children": [
                        {"element": "v-text-field", "props": {"label": "LLM Base URL", "model": "llm_base", "placeholder": "https://api.gptapi.us/v1"}},
                        {"element": "v-text-field", "props": {"label": "LLM Model", "model": "llm_model", "placeholder": "deepseek-v3"}},
                        {"element": "v-text-field", "props": {"label": "LLM API Key", "model": "llm_key", "type": "password"}}
                    ]},
                    {"element": "v-col", "props": {"cols": 12, "md": 6}, "children": [
                        {"element": "v-select", "props": {"label": "AI触发模式", "model": "ask_mode",
                            "items": [{"title": "智能(smart)", "value": "smart"},
                                      {"title": "总是(always)", "value": "always"},
                                      {"title": "手动(manual)", "value": "manual"}]}},
                        {"element": "v-switch", "props": {"label": "自动下载（≥自动阈值）", "model": "auto_download"}},
                        {"element": "v-text-field", "props": {"label": "自动通过阈值", "model": "threshold_auto", "type": "number"}},
                        {"element": "v-text-field", "props": {"label": "人工队列阈值（下限）", "model": "threshold_manual", "type": "number"}}
                    ]}
                ]},
                {"element": "v-alert", "props": {"type": "info", "text": True,
                    "children": ["打分：ID命中/相似度/年份/季集/一致性/加成 + AI贡献；分数可 >100。推荐阈值：自动≥120；人工80–119；<80强制人工。"]}}
            ]
        }

    def get_state(self) -> Optional[dict]:
        return self._cfg

    def set_state(self, state: dict):
        if not state: return
        self._cfg.update(state or {})

    # ===== NameRecognize：只在需要时问 AI =====
    @eventmanager.register(ChainEventType.NameRecognize)
    def on_name_recognize(self, event: Event):
        if not self.is_enabled:
            return
        data = getattr(event, "event_data", None)
        if not data:
            return
        title: str = getattr(data, "title", "") or ""
        subtitle: str = getattr(data, "subtitle", "") or ""
        # 智能决定是否问AI
        mode = self._cfg.get("ask_mode", "smart")
        need_ai = (mode == "always")

        # 判断模糊度（可选后端直连）
        ambiguous = True
        mp_ctx = None
        if self._cfg.get("mp_api_base"):
            mp = MPClient(self._cfg["mp_api_base"], self._cfg["mp_api_token"])
            mp_ctx = mp.recognize(title, subtitle)
            try:
                cands = (mp_ctx or {}).get("candidates") or (mp_ctx or {}).get("data") or []
                ambiguous = len(cands) != 1
            except Exception:
                ambiguous = True

        if mode == "smart" and ambiguous:
            need_ai = True
        if mode == "manual":
            need_ai = False

        if not need_ai:
            return

        # 调 AI
        llm = LLMClient(self._cfg["llm_base"], self._cfg["llm_key"], self._cfg["llm_model"])
        system_prompt = (
            "你是严格模式的命名解析器。只输出一个JSON对象，不得包含解释、Markdown或代码块。"
            "字段：name, version, part, year, resolution, season, episode。"
            "规则：year为4位数字或null；season/episode为正整数或null；其余为字符串或null。"
            "无法解析时输出{}。示例："
            "{\"name\":\"xxx\",\"year\":\"2024\",\"season\":1,\"episode\":2,\"version\":null,\"part\":null,\"resolution\":\"1080p\"}"
        )
        ai = llm.parse_title(title, system_prompt)

        # 组装 ext（可扩展Trakt/TMDB/Bangumi，这里用mp_ctx）
        ext: Dict[str, Any] = {}
        if mp_ctx:
            cands = mp_ctx.get("candidates") or mp_ctx.get("data") or []
            if cands:
                top = cands[0]
                ext["tmdbid"] = top.get("tmdbid")
                ext["doubanid"] = top.get("doubanid")
                ext["bangumiid"] = top.get("bangumiid")
                ext["traktid"] = top.get("traktid")
                ext["imdbid"] = top.get("imdbid")
                ext["year"] = top.get("year")
                ext["se"] = {"season": top.get("season"), "episode": top.get("episode")}
                names = []
                for k in ("name", "title", "cn_name", "jp_name", "en_name"):
                    if top.get(k): names.append(str(top[k]))
                ext["names"] = list(set(names))
                ext["is_anime"] = (top.get("type") == "anime")
                ext["is_movie"] = (top.get("type") == "movie")

        # 打分
        scorer = Scorer(self._cfg.get("weights") or {})
        bd = scorer.score(title, ai or {}, ext or {})
        total = bd.total
        th_auto = int(self._cfg.get("threshold_auto", THRESHOLD_AUTO_DEFAULT))
        th_manual = int(self._cfg.get("threshold_manual", THRESHOLD_MANUAL_DEFAULT))

        # 自动下载（可选）
        if total >= th_auto and self._cfg.get("auto_download"):
            body = {
                "mediainfo": {
                    "title": (ai or {}).get("name") or title,
                    "year": (ai or {}).get("year") or ext.get("year"),
                    "season": _safe_int((ai or {}).get("season")),
                    "episode": _safe_int((ai or {}).get("episode")),
                    "tmdbid": ext.get("tmdbid"),
                    "doubanid": ext.get("doubanid"),
                    "bangumiid": ext.get("bangumiid"),
                    "traktid": ext.get("traktid"),
                }
            }
            if self._cfg.get("mp_api_base") and self._cfg.get("mp_api_token"):
                resp = MPClient(self._cfg["mp_api_base"], self._cfg["mp_api_token"]).download_with_media(body)
                logger.info(f"[MSAIR] auto download resp: {resp}")
            else:
                # 无直连：放队列，前端可一键提交到 api/v1/download/
                iid = _gen_id()
                with self._lock:
                    self._queue[iid] = {
                        "id": iid, "title": title, "ai": ai or {}, "ext": ext or {},
                        "score": {"total": total, "items": bd.items},
                        "auto_download_payload": body
                    }
            return

        # 入人工队列 / 或低分直接放过
        if total >= th_manual:
            iid = _gen_id()
            with self._lock:
                self._queue[iid] = {
                    "id": iid, "title": title, "ai": ai or {}, "ext": ext or {},
                    "score": {"total": total, "items": bd.items}
                }
        # < th_manual ：不处理（交由其它插件/主程序）

    # ===== 页面：人工队列 + 批量AI + 🧪自检 =====
    def get_page(self) -> Optional[dict]:
        rows = []
        with self._lock:
            for k, v in self._queue.items():
                rows.append({
                    "id": k,
                    "title": v.get("title"),
                    "name": (v.get("ai") or {}).get("name"),
                    "year": (v.get("ai") or {}).get("year"),
                    "score": (v.get("score") or {}).get("total", 0),
                })

        # 自检日志展示
        logs_block = []
        if self._selftest:
            txt = "\n".join(self._selftest.get("logs", [])) or "（无日志）"
            ok = bool(self._selftest.get("ok"))
            logs_block = [{
                "element": "v-alert",
                "props": {"type": "success" if ok else "error", "text": True},
                "children": [f"🧪 自检结果：{'通过' if ok else '存在问题'}"],
            }, {
                "element": "v-card", "children": [
                    {"element": "v-card-title", "children": ["自检日志"]},
                    {"element": "v-card-text", "children": [txt]}
                ]
            }]

        return {
            "element": "v-container",
            "props": {"fluid": True},
            "children": [
                {"element": "v-row", "children": [
                    {"element": "v-col", "props": {"cols": 12, "md": 6}, "children": [
                        {"element": "v-btn", "props": {"color": "primary", "class": "mr-2"},
                         "events": {"click": {"api": "plugin/multisource_ai_recognizer/ai_batch", "method": "post", "json": {"scope": "all"}}},
                         "children": ["🤖 AI识别（全部）"]},
                        {"element": "v-btn", "props": {"color": "primary", "class": "mr-2"},
                         "events": {"click": {"api": "plugin/multisource_ai_recognizer/ai_batch", "method": "post", "json": {"scope": "selected", "ids": "{{selectedIds}}"}}},
                         "children": ["🤖 AI识别（所选）"]},
                        {"element": "v-btn", "props": {"color": "secondary"},
                         "events": {"click": {"api": "plugin/multisource_ai_recognizer/selftest", "method": "post"}},
                         "children": ["🧪 自检"]}
                    ]},
                    {"element": "v-col", "props": {"cols": 12, "md": 6}, "children": [
                        {"element": "v-select", "props": {
                            "label": "AI触发模式", "model": "ask_mode",
                            "items": [{"title": "智能(smart)", "value": "smart"},
                                      {"title": "总是(always)", "value": "always"},
                                      {"title": "手动(manual)", "value": "manual"}],
                            "value": self._cfg.get("ask_mode", "smart")
                        },
                         "events": {"change": {"api": "plugin/multisource_ai_recognizer/config", "method": "post", "json": {"ask_mode": "{{ask_mode}}"}}}
                        }
                    ]}
                ]},
                # 目标目录输入（稳妥，不依赖弹窗）
                {"element": "v-row", "children": [
                    {"element": "v-col", "props": {"cols": 12, "md": 4}, "children": [
                        {"element": "v-text-field", "props": {"label": "目标存储类型", "model": "target_storage", "placeholder": "local/nas/..."}}
                    ]},
                    {"element": "v-col", "props": {"cols": 12, "md": 8}, "children": [
                        {"element": "v-text-field", "props": {"label": "目标目录路径", "model": "target_path", "placeholder": "/media/Movies"}}
                    ]}
                ]},
                # 队列表
                {"element": "v-data-table",
                 "props": {
                    "items": rows,
                    "show-select": True,
                    "showSelect": True,
                    "model": "selectedIds",
                    "headers": [
                        {"title": "ID", "value": "id"},
                        {"title": "标题", "value": "title"},
                        {"title": "名称", "value": "name"},
                        {"title": "年份", "value": "year"},
                        {"title": "得分", "value": "score"},
                        {"title": "操作", "value": "actions"}
                    ],
                    "item": {
                        "actions": {
                            "element": "v-btn",
                            "props": {"text": True, "color": "primary"},
                            "events": {"click": {
                                "api": "plugin/multisource_ai_recognizer/confirm",
                                "method": "post",
                                "json": {"id": "{{item.id}}",
                                         "target_storage": "{{target_storage}}",
                                         "target_path": "{{target_path}}"}
                            }},
                            "children": ["确认并整理"]
                        }
                    }
                 }
                },
                # 自检日志块（如有）
                *logs_block
            ]
        }

    # ===== 页面 API =====
    def get_api(self) -> Optional[List[dict]]:
        return [
            {"path": "/queue", "endpoint": self.api_queue, "methods": ["GET"], "summary": "获取人工队列"},
            {"path": "/confirm", "endpoint": self.api_confirm, "methods": ["POST"], "summary": "确认并整理"},
            {"path": "/config", "endpoint": self.api_config, "methods": ["POST"], "summary": "更新配置（ask_mode/阈值）"},
            {"path": "/ai_batch", "endpoint": self.api_ai_batch, "methods": ["POST"], "summary": "批量AI识别（全部/所选）"},
            {"path": "/selftest", "endpoint": self.api_selftest, "methods": ["POST"], "summary": "插件自检（LLM/MP/评分/流程）"},
        ]

    def api_queue(self):
        with self._lock:
            return {"data": list(self._queue.values())}

    def api_config(self, **kwargs):
        changed = {}
        for k in ("ask_mode", "auto_download", "threshold_auto", "threshold_manual"):
            if k in kwargs:
                self._cfg[k] = kwargs[k]
                changed[k] = kwargs[k]
        return {"code": 0, "msg": "ok", "changed": changed}

    def api_ai_batch(self, scope: str = "all", ids: Optional[List[str]] = None):
        # 遍历目标项，对“还没有 AI 结果”的再问一次 AI（节省费用）
        cnt = 0
        targets: List[str] = []
        with self._lock:
            if scope == "selected" and ids:
                targets = [i for i in ids if i in self._queue]
            else:
                targets = list(self._queue.keys())
        for iid in targets:
            with self._lock:
                item = self._queue.get(iid)
            if not item:
                continue
            title = item.get("title") or ""
            if (item.get("ai") or {}).get("name"):
                continue
            llm = LLMClient(self._cfg["llm_base"], self._cfg["llm_key"], self._cfg["llm_model"])
            system_prompt = (
                "你是严格模式的命名解析器。只输出一个JSON对象，不得包含解释、Markdown或代码块。"
                "字段：name, version, part, year, resolution, season, episode。"
                "规则：year为4位数字或null；season/episode为正整数或null；其余为字符串或null。"
                "无法解析时输出{}。示例："
                "{\"name\":\"xxx\",\"year\":\"2024\",\"season\":1,\"episode\":2,\"version\":null,\"part\":null,\"resolution\":\"1080p\"}"
            )
            ai = llm.parse_title(title, system_prompt)
            if ai:
                with self._lock:
                    item["ai"] = ai
                    cnt += 1
        return {"code": 0, "msg": f"AI识别完成：{cnt} 条"}

    def api_confirm(self, id: str, target_storage: str = "", target_path: str = "", background: bool = True):
        with self._lock:
            item = self._queue.get(id)
        if not item:
            return {"code": 404, "msg": "not found"}

        body = {
            "items": [{
                "path": "",  # 若来源于本地文件识别，可放真实路径
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
                }
            }],
            "background": background
        }

        if self._cfg.get("mp_api_base") and self._cfg.get("mp_api_token"):
            resp = MPClient(self._cfg["mp_api_base"], self._cfg["mp_api_token"]).transfer_manual(body)
            with self._lock:
                self._queue.pop(id, None)
            return {"code": 0, "msg": "已提交整理", "resp": resp}
        else:
            with self._lock:
                self._queue.pop(id, None)
            return {"next_api": "api/v1/transfer/manual", "method": "post", "json": body}

    def api_selftest(self):
        """
        一键自检：返回结构 + 写入 self._selftest（页面刷新显示）
        """
        logs: List[str] = []
        ok = True
        t0 = time.time()

        def log(s, good=None):
            flag = "✓" if good is True else ("✗" if good is False else "•")
            msg = f"{flag} {s}"
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
            log("主程序后端直连未配置（mp_api_base/mp_api_token），跳过直连自测（前端相对路由不受影响）。")
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
            ai = {"name": "Demo Title", "year": "2024", "season": 1, "episode": 1, "resolution": "1080p", "version": None, "part": None}
            ext = {"names": ["Demo Title", "演示标题"], "year": "2024", "se": {"season": 1, "episode": 1}, "tmdbid": 1, "is_movie": True, "imdbid": "tt123", "agree_pairs": 2}
            bd = scorer.score("Demo Title.2024.S01E01.1080p", ai, ext)
            if isinstance(bd.total, int):
                log(f"打分器运行正常，总分={bd.total}。", True)
            else:
                log("打分器返回异常类型。", False); ok = False
        except Exception as e:
            log(f"打分器异常：{e}", False); ok = False

        # 4) 队列与确认流程（不真正提交）
        try:
            iid = _gen_id()
            with self._lock:
                self._queue[iid] = {
                    "id": iid,
                    "title": "SelfTest Item 2024 S01E01 1080p",
                    "ai": {"name": "SelfTest Item", "year": "2024", "season": 1, "episode": 1},
                    "ext": {"tmdbid": 1},
                    "score": {"total": 130, "items": [("demo", 130)]}
                }
            r = self.api_confirm(iid, target_storage="local", target_path="/media/SelfTest", background=True)
            # 如果未配置 mp 直连，会返回 next_api 让前端继续；这也算通过
            if isinstance(r, dict) and (r.get("code") == 0 or r.get("next_api")):
                log("确认流程可用（生成 transfer/manual payload 或已提交）。", True)
            else:
                log("确认流程返回异常。", False); ok = False
        except Exception as e:
            log(f"确认流程异常：{e}", False); ok = False

        cost = round((time.time() - t0)*1000)
        log(f"自检完成，用时 {cost} ms。")
        self._selftest = {"ok": ok, "logs": logs, "ts": int(time.time())}
        return {"code": 0 if ok else 1, "msg": "done", "data": self._selftest}

    # ===== 仪表板 =====
    def get_dashboard_meta(self) -> Optional[List[Dict[str, str]]]:
        return [{"key": "queue", "name": "多源AI识别：待确认"}]

    def get_dashboard(self, key: str, **kwargs) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], List[dict]]]:
        if key != "queue":
            return None
        cols = {"cols": 12, "md": 6}
        conf = {"refresh": 10, "title": "AI待确认列表"}
        with self._lock:
            top = [{"id": k, "title": v.get("title"), "score": (v.get("score") or {}).get("total", 0)}
                   for k, v in list(self._queue.items())[:6]]
        page = [{
            "element": "v-list", "children": [
                {"element": "v-list-item",
                 "props": {"title": "{{item.title}}", "subtitle": "分数：{{item.score}}"},
                 "for": {"item": top},
                 "children": [{
                     "element": "v-btn", "props": {"text": True, "size": "small"},
                     "events": {"click": {"api": "plugin/multisource_ai_recognizer/confirm",
                                          "method": "post",
                                          "json": {"id": "{{item.id}}", "target_storage": "local", "target_path": "/media/Movies"}}},
                     "children": ["快速确认"]
                 }]}
            ]
        }]
        return cols, conf, page

    # ===== 工作流（v2.4.8+） =====
    def get_actions(self) -> List[Dict[str, Any]]:
        return [{"id": "msair_recognize", "name": "多源AI识别评分", "func": self.action_ai_recognize, "kwargs": {}}]

    def action_ai_recognize(self, action_content, **kwargs):
        title = getattr(action_content, "title", None) or getattr(action_content, "name", None)
        if not title:
            return False, action_content
        llm = LLMClient(self._cfg["llm_base"], self._cfg["llm_key"], self._cfg["llm_model"])
        system_prompt = (
            "你是严格模式的命名解析器。只输出一个JSON对象，不得包含解释、Markdown或代码块。"
            "字段：name, version, part, year, resolution, season, episode。"
            "规则：year为4位数字或null；season/episode为正整数或null；其余为字符串或null。"
            "无法解析时输出{}。示例："
            "{\"name\":\"xxx\",\"year\":\"2024\",\"season\":1,\"episode\":2,\"version\":null,\"part\":null,\"resolution\":\"1080p\"}"
        )
        ai = llm.parse_title(title, system_prompt)
        bd = Scorer(self._cfg.get("weights") or {}).score(title, ai or {}, {})
        setattr(action_content, "ext", {"ai": ai, "score": {"total": bd.total, "items": bd.items}})
        return True, action_content

    # ===== 消息交互（v2.5.7+） =====
    def get_command(self) -> List[Dict[str, Any]]:
        return [{
            "cmd": "/msair",
            "event": EventType.PluginAction,
            "desc": "多源AI识别面板",
            "category": "插件交互",
            "data": {"action": "msair_menu"}
        }]

    @eventmanager.register(EventType.PluginAction)
    def command_action(self, event: Event):
        data = getattr(event, "event_data", {}) or {}
        if data.get("action") != "msair_menu":
            return
        channel = data.get("channel"); user = data.get("user")
        buttons = [[
            {"text": "🔍 查看待确认", "callback_data": f"[PLUGIN]{self.__class__.__name__}|queue"},
            {"text": "⚙️ 设置", "callback_data": f"[PLUGIN]{self.__class__.__name__}|settings"}
        ]]
        self.post_message(channel=channel, title="多源AI识别", text="请选择：", userid=user, buttons=buttons)

    @eventmanager.register(EventType.MessageAction)
    def message_action(self, event: Event):
        data = getattr(event, "event_data", {}) or {}
        if data.get("plugin_id") != self.__class__.__name__:
            return
        text = data.get("text", "")
        channel = data.get("channel"); user = data.get("userid")
        if text == "queue":
            with self._lock:
                cnt = len(self._queue)
            self.post_message(channel=channel, title="队列", text=f"当前待确认 {cnt} 条", userid=user)
        elif text == "settings":
            self.post_message(channel=channel, title="设置", text="请在插件页面中修改配置。", userid=user)

    # ===== 存储扩展骨架（可留空） =====
    def get_module(self) -> Dict[str, Any]:
        return {
            "list_files": self.list_files,
            "any_files": self.any_files,
            "download_file": self.download_file,
            "upload_file": self.upload_file,
            "delete_file": self.delete_file,
            "rename_file": self.rename_file,
            "get_file_item": self.get_file_item,
            "get_parent_item": self.get_parent_item,
            "snapshot_storage": self.snapshot_storage,
            "storage_usage": self.storage_usage,
            "support_transtype": self.support_transtype
        }

    # 占位：仅当 storage=="msair" 时你再改成真实逻辑；否则返回 None
    def list_files(self, fileitem, recursion: bool = False): return None
    def any_files(self, fileitem, extensions: list = None): return None
    def download_file(self, fileitem, path=None): return None
    def upload_file(self, fileitem, path, new_name: Optional[str] = None): return None
    def delete_file(self, fileitem): return None
    def rename_file(self, fileitem, name: str): return None
    def get_file_item(self, storage: str, path): return None
    def get_parent_item(self, fileitem): return None
    def snapshot_storage(self, storage: str, path): return None
    def storage_usage(self, storage: str): return None
    @staticmethod
    def support_transtype(storage: str) -> Optional[dict]:
        return {"move": "移动", "copy": "复制"}
