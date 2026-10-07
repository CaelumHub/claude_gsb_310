"""构建对比：两场构建的用例集合对齐与差异分析。

「这次比上次好了还是差了」看似简单，实际有三个坑：

1. **用例集合不一定相同**：两场构建可能跑不同套件，或期间有用例增删。
   逐条按下标 / 按名字比对，会漏掉「只在其中一场存在」的用例；
2. **同名 ≠ 同一条用例**：用例可能被删除重建（id 变了名字没变），按名字
   硬合并会把两条不同用例的结果错配到一起，得出错误的「新增失败 /
   已修复」结论；
3. **跨环境对比**：同一用例在不同环境（延迟、失败率、变量不同）下，耗时
   与结果本来就不同。不区分「环境差异」与「真实退化」，会把环境噪声
   当成回归报警。

本模块的对齐与判定策略
----------------------
- **对齐键**：优先 ``case_id``（用例实体的稳定 id）；结果缺 id 时退化为
  「在两边都唯一的用例名」对齐，重名则放弃对齐；
- **同名不同 id**：不自动合并，列入 ``ambiguous`` 并说明原因，同时保留在
  「仅单侧存在」清单里——不丢数据，也不硬比出错误结论；
- **双口径通过率**：整体通过率（集合不同，仅供参考）+ 共同用例子集通过率
  （严格可比口径），结论以后者为准；
- **跨环境 / 跨套件**：置信度降级（high / medium / low），产出环境配置
  差异表与注意事项，让「环境差异」可见，而不是混进「退化」里；
- **无法对比时给说明**：共同用例为 0 时结论为 ``inconclusive`` 并列出
  原因，绝不硬比一个数出来。
"""

from __future__ import annotations

import statistics
import time
from typing import Optional

from .report import FAILED_STATUSES, _percentile

# 耗时退化判定阈值：绝对差 >= 0.2s 且相对变化 >= 30% 才算「明显变快/变慢」，
# 避免把毫秒级抖动当成退化报警。
MIN_DURATION_DELTA_S = 0.2
MIN_DURATION_DELTA_PCT = 30.0
# 基准耗时为 0（如被跳过）时，只看绝对差，放宽到 0.5s
MIN_DURATION_DELTA_WHEN_BASE_ZERO_S = 0.5

# 耗时分布分桶边界（秒）
_DURATION_BUCKET_EDGES = (0.1, 0.5, 1.0, 3.0, 10.0)
_DURATION_BUCKET_LABELS = ("<0.1s", "0.1~0.5s", "0.5~1s", "1~3s", "3~10s", ">10s")

_TOP_N = 10


def _status_group(status: str) -> str:
    """把用例状态归并到 passed / failed / skipped 三组用于迁移分类。"""
    if status == "passed":
        return "passed"
    if status in FAILED_STATUSES:
        return "failed"
    return "skipped"


def _failure_reason(result: dict) -> str:
    """从失败断言 → 失败步骤 → 结果消息中提取失败原因。"""
    step = next((s for s in result.get("steps", [])
                 if s.get("status") in ("failed", "error")), None)
    assertion = next((a for a in result.get("assertions", []) if not a.get("ok")), None)
    return ((assertion or {}).get("message") or (step or {}).get("message")
            or result.get("message") or "")


def _duration_stats(durations: list[float]) -> dict:
    if not durations:
        return {"avg": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0,
                "max": 0.0, "min": 0.0}
    return {
        "avg": round(statistics.mean(durations), 3),
        "median": round(statistics.median(durations), 3),
        "p95": _percentile(durations, 0.95),
        "p99": _percentile(durations, 0.99),
        "max": round(max(durations), 3),
        "min": round(min(durations), 3),
    }


def _duration_buckets(durations: list[float]) -> list[int]:
    counts = [0] * (len(_DURATION_BUCKET_EDGES) + 1)
    for d in durations:
        i = 0
        while i < len(_DURATION_BUCKET_EDGES) and d >= _DURATION_BUCKET_EDGES[i]:
            i += 1
        counts[i] += 1
    return counts


def _align(base_results: list[dict], target_results: list[dict]):
    """对齐两场构建的用例集合。

    返回 ``(matched, only_base, only_target, ambiguous)``：

    - ``matched``     共同用例 ``[(key, base_result, target_result)]``，逐项可比；
    - ``only_base`` / ``only_target``  仅单侧存在的用例，单独列出、不参与逐项对比；
    - ``ambiguous``   同名但 id 不同的用例组，无法确认是否同一条用例，不自动合并。
    """
    def _split(results):
        by_id, no_id = {}, []
        for r in results:
            cid = r.get("case_id")
            if cid:
                by_id.setdefault(cid, r)
            else:
                no_id.append(r)
        return by_id, no_id

    base_by_id, base_no_id = _split(base_results)
    target_by_id, target_no_id = _split(target_results)

    matched: list[tuple[str, dict, dict]] = []
    only_base: list[dict] = []
    only_target: list[dict] = []

    # 1) 主键对齐：case_id 相同即同一条用例
    for cid, b in base_by_id.items():
        t = target_by_id.get(cid)
        if t is not None:
            matched.append((cid, b, t))
        else:
            only_base.append(b)
    for cid, t in target_by_id.items():
        if cid not in base_by_id:
            only_target.append(t)

    # 2) 缺 id 的结果：名字在两边都唯一时才敢对齐，重名宁可放弃
    def _unique_by_name(records):
        counts: dict[str, int] = {}
        for r in records:
            name = r.get("case_name") or ""
            counts[name] = counts.get(name, 0) + 1
        return {r.get("case_name"): r for r in records
                if r.get("case_name") and counts[r.get("case_name")] == 1}

    base_unique = _unique_by_name(base_no_id)
    target_unique = _unique_by_name(target_no_id)
    matched_names = {n for n in base_unique if n in target_unique}
    for name in sorted(matched_names):
        matched.append((f"name:{name}", base_unique[name], target_unique[name]))
    for r in base_no_id:
        if r.get("case_name") not in matched_names:
            only_base.append(r)
    for r in target_no_id:
        if r.get("case_name") not in matched_names:
            only_target.append(r)

    # 3) 同名不同 id：提示「疑似同一用例」，但不合并（可能含义已不同）
    def _names_of(records):
        out: dict[str, list] = {}
        for r in records:
            out.setdefault(r.get("case_name") or "(未命名)", []).append(r.get("case_id"))
        return out

    base_names = _names_of(only_base)
    target_names = _names_of(only_target)
    ambiguous = [{
        "name": name,
        "base_case_ids": [c for c in base_names[name] if c],
        "target_case_ids": [c for c in target_names[name] if c],
        "reason": "同名但用例 id 不同，可能是删除重建或复制的用例，未自动合并对比",
    } for name in sorted(set(base_names) & set(target_names))]

    return matched, only_base, only_target, ambiguous


class BuildComparator:
    """两场构建的对比分析器。"""

    def __init__(self, build_registry, store_registry=None,
                 env_manager=None, coverage=None):
        self.builds = build_registry
        self.stores = store_registry
        self.env_manager = env_manager
        self.coverage = coverage

    # ------------------------------------------------------------------ 主入口
    def compare(self, base_id: str, target_id: str) -> dict:
        """对比两场构建，返回完整的对比报告（或 ``{"error": ...}``）。"""
        base = self.builds.find_build(base_id)
        if base is None:
            return {"error": f"基准构建不存在: {base_id}"}
        target = self.builds.find_build(target_id)
        if target is None:
            return {"error": f"目标构建不存在: {target_id}"}

        base_results = self.builds.for_project(base["project_id"]).results(base_id)
        target_results = self.builds.for_project(target["project_id"]).results(target_id)

        matched, only_base, only_target, ambiguous = _align(base_results, target_results)
        classified = self._classify(matched)

        same_project = base["project_id"] == target["project_id"]
        same_env = base.get("env_id") == target.get("env_id")
        same_suite = base.get("suite_id") == target.get("suite_id")

        env_diff = self._env_diff(base, target)
        base_rate, target_rate = self._matched_pass_rates(matched)
        summary = {
            "base": self._summary_of(base),
            "target": self._summary_of(target),
            "matched": {
                "count": len(matched),
                "base_pass_rate": base_rate,
                "target_pass_rate": target_rate,
                "pass_rate_delta": round(target_rate - base_rate, 1),
            },
        }
        summary["delta"] = {
            "pass_rate": round(summary["target"]["pass_rate"] - summary["base"]["pass_rate"], 1),
            "duration": round(summary["target"]["duration"] - summary["base"]["duration"], 3),
            "total": summary["target"]["total"] - summary["base"]["total"],
        }

        caveats = self._caveats(base, target, same_project, same_env, same_suite,
                                only_base, only_target, ambiguous, env_diff)
        confidence = self._confidence(same_project, same_env, same_suite)

        return {
            "base": self._build_ref(base),
            "target": self._build_ref(target),
            "same_project": same_project,
            "same_env": same_env,
            "same_suite": same_suite,
            "confidence": confidence,
            "verdict": self._verdict(classified, len(matched), base_rate, target_rate),
            "caveats": caveats,
            "summary": summary,
            "cases": {
                "new_failures": classified["new_failures"],
                "fixed": classified["fixed"],
                "still_failing": classified["still_failing"],
                "other_changes": classified["other_changes"],
                "stable_count": classified["stable_count"],
                "only_base": [self._case_ref(r) for r in only_base],
                "only_target": [self._case_ref(r) for r in only_target],
                "ambiguous": ambiguous,
            },
            "durations": self._durations(base, target, matched,
                                         env_influenced=not same_env),
            "coverage": self._coverage_diff(base, target),
            "env_diff": env_diff,
            "links": {
                "base": self._links(base),
                "target": self._links(target),
            },
            "generated_at": time.time(),
        }

    # ------------------------------------------------------------------ 对齐后分类
    def _classify(self, matched: list) -> dict:
        out = {"new_failures": [], "fixed": [], "still_failing": [],
               "other_changes": [], "stable_count": 0}
        for _key, b, t in matched:
            bg, tg = _status_group(b.get("status")), _status_group(t.get("status"))
            item = self._case_diff(b, t)
            if bg == "passed" and tg == "passed":
                out["stable_count"] += 1
            elif bg == "failed" and tg == "failed":
                item["reason"] = _failure_reason(t) or _failure_reason(b)
                out["still_failing"].append(item)
            elif bg == "passed" and tg == "failed":
                item["reason"] = _failure_reason(t)
                out["new_failures"].append(item)
            elif bg == "failed" and tg == "passed":
                item["reason"] = _failure_reason(b)
                out["fixed"].append(item)
            else:
                # 涉及跳过的迁移（如 skipped→failed）不算「新增失败」，
                # 单独列出并标注前后状态，避免过度解读
                out["other_changes"].append(item)
        for key in ("new_failures", "fixed", "still_failing", "other_changes"):
            out[key].sort(key=lambda x: (x.get("case_name") or "", x.get("case_id") or ""))
        return out

    @staticmethod
    def _case_diff(b: dict, t: dict) -> dict:
        bd, td = b.get("duration") or 0.0, t.get("duration") or 0.0
        return {
            "case_id": t.get("case_id") or b.get("case_id"),
            "case_name": t.get("case_name") or b.get("case_name"),
            "group": t.get("group") or b.get("group"),
            "priority": t.get("priority") or b.get("priority"),
            "base_status": b.get("status"),
            "target_status": t.get("status"),
            "base_duration": bd,
            "target_duration": td,
            "duration_delta": round(td - bd, 3),
            "reason": "",
        }

    @staticmethod
    def _case_ref(r: dict) -> dict:
        return {
            "case_id": r.get("case_id"),
            "case_name": r.get("case_name"),
            "group": r.get("group"),
            "priority": r.get("priority"),
            "status": r.get("status"),
            "duration": r.get("duration") or 0.0,
        }

    @staticmethod
    def _matched_pass_rates(matched: list) -> tuple[float, float]:
        """共同用例子集上的通过率（与报告口径一致：跳过不计入分母）。"""
        def _rate(side: int) -> float:
            finished = passed = 0
            for _, b, t in matched:
                r = b if side == 0 else t
                if _status_group(r.get("status")) == "skipped":
                    continue
                finished += 1
                if r.get("status") == "passed":
                    passed += 1
            return round(passed / finished * 100, 1) if finished else 0.0
        return _rate(0), _rate(1)

    # ------------------------------------------------------------------ 汇总与结论
    @staticmethod
    def _summary_of(build: dict) -> dict:
        total = build.get("total", 0)
        skipped = build.get("skipped", 0)
        finished = total - skipped
        passed = build.get("passed", 0)
        return {
            "total": total,
            "passed": passed,
            "failed": build.get("failed", 0),
            "error": build.get("error", 0),
            "timeout": build.get("timeout", 0),
            "skipped": skipped,
            "pass_rate": round(passed / finished * 100, 1) if finished else 0.0,
            "duration": build.get("duration", 0.0),
        }

    @staticmethod
    def _verdict(classified: dict, matched_count: int,
                 base_rate: float, target_rate: float) -> dict:
        if matched_count == 0:
            return {
                "level": "inconclusive",
                "label": "无法直接对比",
                "reasons": ["两场构建没有共同用例（用例集合无交集），无法逐项对比；"
                            "请分别查看两侧构建详情，或选择用例集合有重叠的构建"],
            }
        nf = len(classified["new_failures"])
        fx = len(classified["fixed"])
        still = len(classified["still_failing"])
        delta = round(target_rate - base_rate, 1)
        reasons = []
        if nf:
            reasons.append(f"新增失败 {nf} 条")
        if fx:
            reasons.append(f"已修复 {fx} 条")
        if still:
            reasons.append(f"持续失败 {still} 条")
        reasons.append(f"共同用例通过率 {base_rate}% → {target_rate}%（{delta:+.1f}）")
        if nf and fx:
            level, label = "mixed", "有进有退"
        elif nf:
            level, label = "regression", "出现退化"
        elif fx:
            level, label = "improved", "有所改善"
        else:
            level, label = "stable", "基本持平"
        return {"level": level, "label": label, "reasons": reasons}

    @staticmethod
    def _confidence(same_project: bool, same_env: bool, same_suite: bool) -> str:
        if not same_project or not same_env:
            return "low"
        if not same_suite:
            return "medium"
        return "high"

    def _caveats(self, base: dict, target: dict, same_project: bool,
                 same_env: bool, same_suite: bool, only_base: list,
                 only_target: list, ambiguous: list, env_diff: dict) -> list[str]:
        caveats = []
        if not same_project:
            caveats.append("两场构建属于不同项目，用例与覆盖率口径不同，对比仅供参考")
        if not same_env:
            def _env_label(side: str, env_id) -> str:
                info = env_diff.get(side) or {}
                name = info.get("name") or env_id or "未知环境"
                return f"{name}（已删除）" if info.get("missing") else name
            b_name = _env_label("base_env", base.get("env_id"))
            t_name = _env_label("target_env", target.get("env_id"))
            caveats.append(
                f"运行环境不同（{b_name} → {t_name}）：耗时与结果差异可能来自环境配置，"
                "请结合下方「环境差异」判断，真实退化需在相同环境复跑确认")
        if not same_suite:
            caveats.append("两场构建来自不同套件，用例集合不同：逐项对比仅覆盖共同用例，"
                           "单侧独有的用例已单独列出")
        if only_base or only_target:
            caveats.append(
                f"用例集合不完全相同：仅基准存在 {len(only_base)} 条、仅目标存在 "
                f"{len(only_target)} 条，这些用例不参与逐项对比")
        if ambiguous:
            caveats.append(f"发现 {len(ambiguous)} 组同名但 id 不同的用例，"
                           "可能是删除重建，未自动合并")
        for side, build in (("基准", base), ("目标", target)):
            if build.get("status") in ("pending", "running"):
                caveats.append(f"{side}构建仍在运行中，对比结果会随执行变化")
        return caveats

    # ------------------------------------------------------------------ 耗时
    def _durations(self, base: dict, target: dict, matched: list,
                   env_influenced: bool) -> dict:
        base_durations = base.get("durations", [])
        target_durations = target.get("durations", [])
        slower, faster = self._duration_changes(matched)
        return {
            "base": _duration_stats(base_durations),
            "target": _duration_stats(target_durations),
            "buckets": [
                {"label": label, "base": b, "target": t}
                for label, b, t in zip(_DURATION_BUCKET_LABELS,
                                       _duration_buckets(base_durations),
                                       _duration_buckets(target_durations))
            ],
            "slower": slower,
            "faster": faster,
            # 跨环境时耗时差异可能主要由环境（延迟配置等）引起，前端据此提示
            "env_influenced": env_influenced,
        }

    def _duration_changes(self, matched: list) -> tuple[list, list]:
        changes = []
        for _, b, t in matched:
            # 任一侧被跳过的用例耗时没有可比性
            if "skipped" in (_status_group(b.get("status")), _status_group(t.get("status"))):
                continue
            bd = b.get("duration") or 0.0
            td = t.get("duration") or 0.0
            delta = round(td - bd, 3)
            if bd > 0:
                pct = delta / bd * 100
                significant = (abs(delta) >= MIN_DURATION_DELTA_S
                               and abs(pct) >= MIN_DURATION_DELTA_PCT)
            else:
                pct = None
                significant = abs(delta) >= MIN_DURATION_DELTA_WHEN_BASE_ZERO_S
            if not significant:
                continue
            item = self._case_diff(b, t)
            item["duration_pct"] = round(pct, 1) if pct is not None else None
            changes.append(item)
        slower = sorted((c for c in changes if c["duration_delta"] > 0),
                        key=lambda c: c["duration_delta"], reverse=True)[:_TOP_N]
        faster = sorted((c for c in changes if c["duration_delta"] < 0),
                        key=lambda c: c["duration_delta"])[:_TOP_N]
        return slower, faster

    # ------------------------------------------------------------------ 覆盖率
    def _coverage_diff(self, base: dict, target: dict) -> Optional[dict]:
        if self.coverage is None:
            return None
        cov_base = self.coverage.get(base["project_id"], base["id"])
        cov_target = self.coverage.get(target["project_id"], target["id"])
        files_base = {f["file"]: f for f in cov_base.get("files", [])}
        files_target = {f["file"]: f for f in cov_target.get("files", [])}
        changes = []
        for path in sorted(set(files_base) | set(files_target)):
            fb, ft = files_base.get(path), files_target.get(path)
            if fb and ft:
                delta = round(ft["percent"] - fb["percent"], 1)
                changes.append({"file": path, "base": fb["percent"],
                                "target": ft["percent"], "delta": delta})
            elif fb:
                changes.append({"file": path, "base": fb["percent"],
                                "target": None, "delta": None})
            else:
                changes.append({"file": path, "base": None,
                                "target": ft["percent"], "delta": None})
        changes.sort(key=lambda c: abs(c["delta"] or 0), reverse=True)
        return {
            "base": {"percent": cov_base.get("percent"),
                     "covered_lines": cov_base.get("covered_lines"),
                     "total_lines": cov_base.get("total_lines")},
            "target": {"percent": cov_target.get("percent"),
                       "covered_lines": cov_target.get("covered_lines"),
                       "total_lines": cov_target.get("total_lines")},
            "delta": round((cov_target.get("percent") or 0) - (cov_base.get("percent") or 0), 1),
            "file_changes": changes[:_TOP_N],
        }

    # ------------------------------------------------------------------ 环境差异
    def _env_diff(self, base: dict, target: dict) -> dict:
        b_env_id, t_env_id = base.get("env_id"), target.get("env_id")
        out = {"same_env": b_env_id == t_env_id,
               "base_env": {"id": b_env_id, "name": None, "missing": False},
               "target_env": {"id": t_env_id, "name": None, "missing": False},
               "items": []}
        if self.env_manager is None:
            return out
        b_env = self.env_manager.get(b_env_id) if b_env_id else None
        t_env = self.env_manager.get(t_env_id) if t_env_id else None
        out["base_env"] = {"id": b_env_id, "name": (b_env or {}).get("name"),
                           "missing": bool(b_env_id) and b_env is None}
        out["target_env"] = {"id": t_env_id, "name": (t_env or {}).get("name"),
                             "missing": bool(t_env_id) and t_env is None}
        if b_env is None or t_env is None:
            if b_env_id != t_env_id:
                out["items"].append({
                    "key": "(环境)", "base": (b_env or {}).get("name") or b_env_id,
                    "target": (t_env or {}).get("name") or t_env_id,
                })
            return out
        items = []
        for key in ("python_version", "base_image"):
            if b_env.get(key) != t_env.get(key):
                items.append({"key": key, "base": b_env.get(key),
                              "target": t_env.get(key)})
        for section in ("config", "variables"):
            b_sec = b_env.get(section) or {}
            t_sec = t_env.get(section) or {}
            for k in sorted(set(b_sec) | set(t_sec)):
                if b_sec.get(k) != t_sec.get(k):
                    items.append({"key": f"{section}.{k}",
                                  "base": b_sec.get(k), "target": t_sec.get(k)})
        out["items"] = items
        return out

    # ------------------------------------------------------------------ 引用与链接
    def _build_ref(self, build: dict) -> dict:
        suite_name = None
        suite_id = build.get("suite_id")
        if suite_id and self.stores is not None:
            suite = self.stores.store("suites").get(suite_id)
            suite_name = (suite or {}).get("name")
        return {
            "id": build["id"],
            "project_id": build["project_id"],
            "name": build.get("name") or build["id"],
            "status": build.get("status"),
            "trigger": build.get("trigger"),
            "suite_id": suite_id,
            "suite_name": suite_name,
            "env_id": build.get("env_id"),
            "started_at": build.get("started_at"),
            "finished_at": build.get("finished_at"),
            "duration": build.get("duration", 0.0),
        }

    @staticmethod
    def _links(build: dict) -> dict:
        pid, bid = build["project_id"], build["id"]
        return {
            "report": f"/page/reports.html?project_id={pid}&build_id={bid}",
            "api_report": f"/api/builds/{bid}/report",
        }
