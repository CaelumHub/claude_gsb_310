"""构建对比：两场构建的并排差异分析。

「这次比上次好了还是差了」听起来简单，实际有三类坑，本模块逐一处理：

1. **用例集合不一致**：两场构建可能跑不同套件（或同一套件但用例增删过），
   直接比全量通过率会把「集合变了」误读成「质量变了」。这里把用例按
   ``case_id`` 对齐成「共同 / 仅基准 / 仅对比」三部分：汇总指标同时给
   「全集口径」与「共同集口径」两套，成败判定只基于共同集；集合之外的
   用例单独列出、给出说明，绝不在不可比的集合上硬算结论；
2. **同名错配**：用例可能被删除后重建（名字相同、id 不同、含义不同），
   按名字配对会把两条不同的用例错当成同一条。这里**只按 case_id 配对**；
   同名不同 id 的用例列入「未配对」并显式说明，交由人判断；
3. **跨环境对比**：同一用例在不同环境（latency / fail_rate / base_url
   不同）下耗时与结果本来就会不同。环境不一致时，耗时差异只展示、
   不定性为「回归」，并把环境配置差异列出来作为可能解释，把「环境差异」
   与「真实退化」区分开。
"""

from __future__ import annotations

import statistics
import time
from typing import Optional

from .report import FAILED_STATUSES, _percentile

# 耗时「变慢 / 变快」判定阈值（仅同环境时启用）：
# 绝对差至少 0.5s，且相对基准变化超过 50%，过滤正常抖动。
_SLOWER_ABS = 0.5
_SLOWER_RATIO = 0.5

# 耗时分布直方图分桶（秒）
_DURATION_BUCKETS = (
    ("<0.1s", 0.0, 0.1),
    ("0.1~0.5s", 0.1, 0.5),
    ("0.5~1s", 0.5, 1.0),
    ("1~5s", 1.0, 5.0),
    ("≥5s", 5.0, None),
)

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

_RUNNING_STATUSES = ("pending", "running")


def _norm_status(status: str) -> str:
    """把用例状态归一化为 pass / fail / skip 三档，便于做状态迁移分析。"""
    if status == "passed":
        return "pass"
    if status in FAILED_STATUSES:
        return "fail"
    return "skip"


def _case_view(result: dict) -> dict:
    """从结果记录中抽取对比所需的字段（含失败原因）。"""
    status = result.get("status", "error")
    reason = ""
    if _norm_status(status) == "fail":
        failing_step = next((s for s in result.get("steps", [])
                             if s.get("status") in ("failed", "error")), None)
        failing_assert = next((a for a in result.get("assertions", [])
                               if not a.get("ok")), None)
        reason = ((failing_assert or {}).get("message")
                  or (failing_step or {}).get("message")
                  or result.get("message") or "")
    return {
        "case_id": result.get("case_id"),
        "case_name": result.get("case_name") or result.get("case_id"),
        "group": result.get("group") or "默认",
        "priority": result.get("priority") or "P3",
        "status": status,
        "duration": round(result.get("duration", 0.0) or 0.0, 3),
        "reason": reason,
    }


class BuildComparator:
    """构建对比器：对齐两场构建的用例集合，计算可解释的并排差异。"""

    def __init__(self, build_registry, registry, coverage_analyzer, env_manager):
        self.builds = build_registry
        self.registry = registry
        self.coverage = coverage_analyzer
        self.envs = env_manager

    # ------------------------------------------------------------------ 主入口
    def compare(self, base_id: str, target_id: str) -> dict:
        """对比两场构建，``base`` 为基准（上次），``target`` 为对比（这次）。"""
        if base_id == target_id:
            return {"error": "请选择两场不同的构建"}
        base = self.builds.find_build(base_id)
        if base is None:
            return {"error": "基准构建不存在"}
        target = self.builds.find_build(target_id)
        if target is None:
            return {"error": "对比构建不存在"}

        base_map = self._index_by_case(base["project_id"], base_id)
        target_map = self._index_by_case(target["project_id"], target_id)

        common_ids = [cid for cid in base_map if cid in target_map]
        only_base_ids = [cid for cid in base_map if cid not in target_map]
        only_target_ids = [cid for cid in target_map if cid not in base_map]

        same_project = base.get("project_id") == target.get("project_id")
        same_suite = bool(base.get("suite_id")) and base.get("suite_id") == target.get("suite_id")
        same_env = bool(base.get("env_id")) and base.get("env_id") == target.get("env_id")

        base_env = self._env_info(base.get("env_id"))
        target_env = self._env_info(target.get("env_id"))
        env_diffs = self._env_diffs(base_env, target_env) if not same_env else []

        transitions = self._transitions(common_ids, base_map, target_map)
        common_base_rate = self._set_pass_rate(common_ids, base_map)
        common_target_rate = self._set_pass_rate(common_ids, target_map)
        collisions = self._name_collisions(only_base_ids, only_target_ids,
                                           base_map, target_map)

        notes, caveats = self._notes(
            base, target, same_project, same_suite, same_env,
            base_env, target_env, env_diffs,
            len(common_ids), len(only_base_ids), len(only_target_ids),
            transitions["counts"]["renamed"], collisions)

        base_brief = self._build_brief(base, base_env)
        target_brief = self._build_brief(target, target_env)

        return {
            "base": base_brief,
            "target": target_brief,
            "comparability": {
                "same_project": same_project,
                "same_suite": same_suite,
                "same_env": same_env,
                "env_diffs": env_diffs,
                "common_count": len(common_ids),
                "only_base_count": len(only_base_ids),
                "only_target_count": len(only_target_ids),
                "fully_comparable": (same_project and same_env
                                     and not only_base_ids and not only_target_ids),
                "notes": notes,
            },
            "verdict": self._verdict(transitions, common_base_rate,
                                     common_target_rate, caveats),
            "summary": {
                "overall": {
                    "base": base_brief["summary"],
                    "target": target_brief["summary"],
                    "delta": self._overall_delta(base_brief["summary"],
                                                 target_brief["summary"]),
                },
                "common": {
                    "count": len(common_ids),
                    "base_pass_rate": common_base_rate,
                    "target_pass_rate": common_target_rate,
                    "pass_rate_delta": (round(common_target_rate - common_base_rate, 1)
                                        if common_base_rate is not None
                                        and common_target_rate is not None else None),
                },
                "durations": {
                    "base": self._duration_stats(base.get("durations", [])),
                    "target": self._duration_stats(target.get("durations", [])),
                    "common": self._common_duration(common_ids, base_map, target_map),
                },
                "coverage": self._coverage_compare(base, target, same_project),
            },
            "transitions": transitions,
            "duration_changes": self._duration_changes(common_ids, base_map,
                                                       target_map, same_env),
            "duration_distribution": {
                "buckets": [b[0] for b in _DURATION_BUCKETS],
                "base": self._distribution(base.get("durations", [])),
                "target": self._distribution(target.get("durations", [])),
            },
            "groups": self._groups_compare(base, target),
            "unmatched": {
                "only_base": self._sorted_cases([base_map[c] for c in only_base_ids]),
                "only_target": self._sorted_cases([target_map[c] for c in only_target_ids]),
                "name_collisions": collisions,
            },
            "generated_at": time.time(),
        }

    # ------------------------------------------------------------------ 用例对齐
    def _index_by_case(self, project_id: str, build_id: str) -> dict:
        """把一场构建的结果按 case_id 索引（重复 id 保留最后一条）。"""
        records = self.builds.for_project(project_id).results(build_id)
        out: dict[str, dict] = {}
        for i, r in enumerate(records):
            cid = r.get("case_id") or f"__no_id_{i}"
            out[cid] = _case_view(r)
        return out

    @staticmethod
    def _sorted_cases(cases: list[dict]) -> list[dict]:
        return sorted(cases, key=lambda c: (_PRIORITY_ORDER.get(c["priority"], 9),
                                            c["case_name"] or ""))

    def _transitions(self, common_ids: list, base_map: dict, target_map: dict) -> dict:
        """共同用例的状态迁移：新增失败 / 已修复 / 持续失败 / 其他变化。"""
        new_failures, fixed, persistent, other = [], [], [], []
        still_passing = still_skipped = renamed = 0
        for cid in common_ids:
            b, t = base_map[cid], target_map[cid]
            is_renamed = b["case_name"] != t["case_name"]
            if is_renamed:
                renamed += 1
            bs, ts = _norm_status(b["status"]), _norm_status(t["status"])
            entry = {
                "case_id": cid,
                "case_name": t["case_name"],
                "base_case_name": b["case_name"],
                "renamed": is_renamed,
                "group": t["group"],
                "priority": t["priority"],
                "base": b,
                "target": t,
                "duration_delta": round(t["duration"] - b["duration"], 3),
            }
            if ts == "fail" and bs != "fail":
                new_failures.append(entry)
            elif bs == "fail" and ts == "pass":
                fixed.append(entry)
            elif bs == "fail" and ts == "fail":
                persistent.append(entry)
            elif bs == "pass" and ts == "pass":
                still_passing += 1
            elif bs == "skip" and ts == "skip":
                still_skipped += 1
            else:
                # pass→skip、skip→pass、fail→skip 等：如实列出，不强行归类
                other.append(entry)

        def _order(e):
            return (_PRIORITY_ORDER.get(e["priority"], 9), -e["target"]["duration"])

        new_failures.sort(key=_order)
        fixed.sort(key=_order)
        persistent.sort(key=_order)
        other.sort(key=_order)
        return {
            "new_failures": new_failures,
            "fixed": fixed,
            "persistent_failures": persistent,
            "other_changes": other,
            "counts": {
                "common": len(common_ids),
                "new_failure": len(new_failures),
                "fixed": len(fixed),
                "persistent_failure": len(persistent),
                "still_passing": still_passing,
                "still_skipped": still_skipped,
                "other": len(other),
                "renamed": renamed,
            },
        }

    @staticmethod
    def _name_collisions(only_base_ids: list, only_target_ids: list,
                         base_map: dict, target_map: dict) -> list[dict]:
        """同名但 id 不同的用例：可能是删除重建，列出但不自动配对。"""
        base_names: dict[str, list] = {}
        for cid in only_base_ids:
            base_names.setdefault(base_map[cid]["case_name"], []).append(cid)
        collisions = []
        for cid in only_target_ids:
            name = target_map[cid]["case_name"]
            for b_cid in base_names.get(name, []):
                collisions.append({"case_name": name,
                                   "base_case_id": b_cid,
                                   "target_case_id": cid})
        return collisions

    # ------------------------------------------------------------------ 指标
    @staticmethod
    def _set_pass_rate(case_ids: list, case_map: dict) -> Optional[float]:
        """指定用例集合的通过率（跳过不计入分母）；无可统计用例时返回 None。"""
        finished = passed = 0
        for cid in case_ids:
            st = _norm_status(case_map[cid]["status"])
            if st == "skip":
                continue
            finished += 1
            if st == "pass":
                passed += 1
        if not finished:
            return None
        return round(passed / finished * 100, 1)

    @staticmethod
    def _overall_delta(base_summary: dict, target_summary: dict) -> dict:
        failed = lambda s: s.get("failed", 0) + s.get("error", 0) + s.get("timeout", 0)  # noqa: E731
        return {
            "pass_rate": round(target_summary.get("pass_rate", 0)
                               - base_summary.get("pass_rate", 0), 1),
            "failed": failed(target_summary) - failed(base_summary),
            "total": target_summary.get("total", 0) - base_summary.get("total", 0),
        }

    @staticmethod
    def _duration_stats(durations: list) -> dict:
        durations = [d for d in durations if isinstance(d, (int, float))]
        if not durations:
            return {"avg": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
        return {
            "avg": round(statistics.mean(durations), 3),
            "median": round(statistics.median(durations), 3),
            "p95": _percentile(durations, 0.95),
            "max": round(max(durations), 3),
            "count": len(durations),
        }

    @staticmethod
    def _common_duration(common_ids: list, base_map: dict, target_map: dict) -> dict:
        """共同用例的耗时对比（同一批用例，耗时差才有意义）。"""
        base_durs = [base_map[c]["duration"] for c in common_ids]
        target_durs = [target_map[c]["duration"] for c in common_ids]
        base_avg = statistics.mean(base_durs) if base_durs else 0.0
        target_avg = statistics.mean(target_durs) if target_durs else 0.0
        return {
            "base_avg": round(base_avg, 3),
            "target_avg": round(target_avg, 3),
            "avg_delta": round(target_avg - base_avg, 3),
        }

    @staticmethod
    def _distribution(durations: list) -> list[int]:
        counts = [0] * len(_DURATION_BUCKETS)
        for d in durations:
            if not isinstance(d, (int, float)):
                continue
            for i, (_, lo, hi) in enumerate(_DURATION_BUCKETS):
                if d >= lo and (hi is None or d < hi):
                    counts[i] += 1
                    break
        return counts

    def _duration_changes(self, common_ids: list, base_map: dict,
                          target_map: dict, same_env: bool) -> dict:
        """共同用例的逐条耗时变化。

        同环境：按阈值判定「变慢 / 变快」；跨环境：环境本身会改变耗时，
        只按差值排序展示，不定性为性能回归。
        """
        rows = []
        for cid in common_ids:
            b, t = base_map[cid], target_map[cid]
            rows.append({
                "case_id": cid,
                "case_name": t["case_name"],
                "group": t["group"],
                "priority": t["priority"],
                "base_duration": b["duration"],
                "target_duration": t["duration"],
                "delta": round(t["duration"] - b["duration"], 3),
            })
        if not same_env:
            rows.sort(key=lambda r: -abs(r["delta"]))
            return {
                "same_env": False,
                "slower": [],
                "faster": [],
                "deltas": rows[:10],
                "note": "两场构建运行环境不同，耗时差异可能来自环境配置（见上方环境差异说明），"
                        "以下仅按差值排序展示，不判定为性能回归。",
            }
        slower, faster = [], []
        for r in rows:
            threshold = max(_SLOWER_ABS, r["base_duration"] * _SLOWER_RATIO)
            if r["delta"] >= threshold:
                slower.append(r)
            elif r["delta"] <= -threshold:
                faster.append(r)
        slower.sort(key=lambda r: -r["delta"])
        faster.sort(key=lambda r: r["delta"])
        return {"same_env": True, "slower": slower[:10], "faster": faster[:10],
                "deltas": [], "note": None}

    def _coverage_compare(self, base: dict, target: dict, same_project: bool) -> dict:
        if not same_project:
            return {"available": False,
                    "note": "两场构建属于不同项目，代码覆盖率口径不同，不做对比。"}
        pid = base["project_id"]
        try:
            b_cov = self.coverage.get(pid, base["id"])
            t_cov = self.coverage.get(pid, target["id"])
        except Exception:  # noqa: BLE001
            return {"available": False, "note": "覆盖率数据不可用。"}
        b_files = {f["file"]: f for f in b_cov.get("files", [])}
        t_files = {f["file"]: f for f in t_cov.get("files", [])}
        changes = []
        for path in sorted(set(b_files) | set(t_files)):
            b, t = b_files.get(path), t_files.get(path)
            if b is None or t is None:
                continue
            delta = round(t["percent"] - b["percent"], 1)
            if delta:
                changes.append({"file": path, "base": b["percent"],
                                "target": t["percent"], "delta": delta})
        changes.sort(key=lambda c: -abs(c["delta"]))
        return {
            "available": True,
            "base": b_cov.get("percent", 0.0),
            "target": t_cov.get("percent", 0.0),
            "delta": round(t_cov.get("percent", 0.0) - b_cov.get("percent", 0.0), 1),
            "files": changes[:12],
        }

    @staticmethod
    def _groups_compare(base: dict, target: dict) -> list[dict]:
        """按分组并排对比；只存在于某一场的分组显式标注，不硬比。"""
        base_groups = base.get("by_group", {}) or {}
        target_groups = target.get("by_group", {}) or {}

        def _rate(entry):
            if not entry:
                return None
            finished = entry.get("total", 0) - entry.get("skipped", 0)
            return round(entry.get("passed", 0) / finished * 100, 1) if finished else None

        out = []
        for g in sorted(set(base_groups) | set(target_groups)):
            b, t = base_groups.get(g), target_groups.get(g)
            out.append({
                "group": g,
                "base": b,
                "target": t,
                "base_pass_rate": _rate(b),
                "target_pass_rate": _rate(t),
                "only": None if (b and t) else ("base" if b else "target"),
            })
        return out

    # ------------------------------------------------------------------ 说明与结论
    def _notes(self, base: dict, target: dict, same_project: bool,
               same_suite: bool, same_env: bool, base_env: dict, target_env: dict,
               env_diffs: list, common: int, only_base: int, only_target: int,
               renamed: int, collisions: list) -> tuple[list, list]:
        """生成可比性说明；``caveats`` 是会并入结论里的简短提醒。"""
        notes, caveats = [], []
        if not same_project:
            notes.append("⚠️ 两场构建属于不同项目，用例、覆盖率口径都不同，"
                         "以下对比仅供参考，建议选择同一项目的构建。")
            caveats.append("跨项目对比，结论仅供参考")
        if not same_suite:
            notes.append(f"两场构建来自不同套件（「{self._suite_name(base.get('suite_id'))}」"
                         f" vs 「{self._suite_name(target.get('suite_id'))}」），用例集合不同："
                         f"共同 {common} 条、仅基准 {only_base} 条、仅对比 {only_target} 条。"
                         f"全量通过率受集合差异影响，请以「共同用例」口径为准。")
        elif only_base or only_target:
            notes.append(f"同一套件但用例集合有变化：仅基准存在 {only_base} 条、"
                         f"仅对比存在 {only_target} 条（可能新增 / 删除了用例），"
                         f"这些用例不参与成败判定，已在下方单独列出。")
        if not same_env:
            notes.append(f"两场构建运行环境不同（「{base_env['name']}」 vs "
                         f"「{target_env['name']}」），结果与耗时差异可能来自环境"
                         f"而非代码变化。")
            caveats.append("跨环境对比，差异可能来自环境")
            if env_diffs:
                notes.append("环境配置差异：" + "；".join(d["text"] for d in env_diffs))
        if renamed:
            notes.append(f"{renamed} 条用例在两场构建之间改过名字，已按用例 id 配对。")
        if collisions:
            notes.append(f"发现 {len(collisions)} 组同名但 id 不同的用例，"
                         f"可能是删除后重建，未自动配对（见「未配对用例」）。")
        if base.get("status") in _RUNNING_STATUSES or target.get("status") in _RUNNING_STATUSES:
            notes.append("有构建尚未结束，对比结果可能不完整。")
        return notes, caveats

    @staticmethod
    def _verdict(transitions: dict, common_base_rate: Optional[float],
                 common_target_rate: Optional[float], caveats: list) -> dict:
        """基于共同用例集合给出「更好 / 更差 / 持平 / 无法比较」的结论。"""
        counts = transitions["counts"]
        if counts["common"] == 0:
            return {
                "code": "incomparable",
                "text": "无法直接比较",
                "reasons": ["两场构建没有共同用例（可能跑了完全不同的套件），"
                            "请对比各自的全量指标，或选择有用例交集的构建。"] + caveats,
            }
        new_f, fixed = counts["new_failure"], counts["fixed"]
        delta = None
        if common_base_rate is not None and common_target_rate is not None:
            delta = round(common_target_rate - common_base_rate, 1)

        reasons = []
        if new_f:
            reasons.append(f"新增失败 {new_f} 条")
        if fixed:
            reasons.append(f"已修复 {fixed} 条")
        if counts["persistent_failure"]:
            reasons.append(f"持续失败 {counts['persistent_failure']} 条")
        if delta is not None:
            reasons.append(f"共同用例通过率 {common_base_rate}% → "
                           f"{common_target_rate}%（{delta:+.1f}）")

        if new_f == 0 and fixed == 0:
            if delta is not None and delta >= 1.0:
                code, text = "improved", "整体更好"
            elif delta is not None and delta <= -1.0:
                code, text = "regressed", "整体变差"
            else:
                code, text = "similar", "基本持平"
        elif new_f == 0:
            code, text = "improved", "整体更好"
        elif new_f > fixed:
            code, text = "regressed", "整体变差"
        elif fixed > new_f:
            code, text = "improved", "整体更好"
        else:
            code, text = "similar", "有进有退，基本持平"
        reasons.extend(caveats)
        return {"code": code, "text": text, "reasons": reasons}

    # ------------------------------------------------------------------ 辅助
    def _env_info(self, env_id: Optional[str]) -> dict:
        env = self.envs.get(env_id) if env_id else None
        if env is None:
            return {"id": env_id, "name": env_id or "未指定", "config": {}}
        return {"id": env_id, "name": env.get("name", env_id),
                "python_version": env.get("python_version"),
                "config": dict(env.get("config") or {})}

    @staticmethod
    def _env_diffs(base_env: dict, target_env: dict) -> list[dict]:
        diffs = []
        for key in sorted(set(base_env["config"]) | set(target_env["config"])):
            a, b = base_env["config"].get(key), target_env["config"].get(key)
            if a != b:
                diffs.append({"key": key, "base": a, "target": b,
                              "text": f"{key}: {a} → {b}"})
        return diffs

    def _suite_name(self, suite_id: Optional[str]) -> str:
        if not suite_id:
            return "未关联套件"
        suite = self.registry.store("suites").get(suite_id)
        return (suite or {}).get("name") or suite_id

    def _build_brief(self, build: dict, env: dict) -> dict:
        finished = build.get("total", 0) - build.get("skipped", 0)
        return {
            "build_id": build["id"],
            "project_id": build.get("project_id"),
            "name": build.get("name") or build["id"],
            "status": build.get("status"),
            "trigger": build.get("trigger"),
            "suite_id": build.get("suite_id"),
            "suite_name": self._suite_name(build.get("suite_id")),
            "env_id": build.get("env_id"),
            "env_name": env["name"],
            "started_at": build.get("started_at"),
            "finished_at": build.get("finished_at"),
            "duration": build.get("duration", 0.0),
            "summary": {
                "total": build.get("total", 0),
                "passed": build.get("passed", 0),
                "failed": build.get("failed", 0),
                "error": build.get("error", 0),
                "skipped": build.get("skipped", 0),
                "timeout": build.get("timeout", 0),
                "pass_rate": round(build.get("passed", 0) / finished * 100, 1) if finished else 0.0,
            },
        }
