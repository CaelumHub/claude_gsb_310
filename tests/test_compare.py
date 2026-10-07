"""构建对比测试。

覆盖：用例集合对齐（共同 / 仅基准 / 仅对比）、状态迁移（新增失败 / 已修复 /
持续失败）、同名不同 id 不错配、跨环境可比性（耗时差异不定性为回归）、
结论判定与 API 路由接线。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (BuildComparator, CoverageAnalyzer, EnvironmentManager,  # noqa: E402
                    new_id)
from storage import BuildStoreRegistry, StoreRegistry  # noqa: E402


def _result(case_id, name, status, duration=0.1, group="g", priority="P2"):
    return {"case_id": case_id, "case_name": name, "group": group,
            "priority": priority, "status": status, "duration": duration,
            "steps": [], "assertions": [], "logs": []}


class _Fixture:
    """一套临时数据目录下的对比器与依赖。"""

    def __init__(self, root):
        self.registry = StoreRegistry(os.path.join(root, "store"))
        self.builds = BuildStoreRegistry(os.path.join(root, "builds"))
        self.env_mgr = EnvironmentManager(self.registry, root)
        self.coverage = CoverageAnalyzer(self.builds)
        self.comparator = BuildComparator(self.builds, self.registry,
                                          self.coverage, self.env_mgr)

    def make_env(self, project_id, name, **config):
        return self.env_mgr.create(project_id, {"name": name, "config": config})

    def make_build(self, project_id, results, env_id=None, suite_id=None,
                   name="构建"):
        bid = new_id("build")
        store = self.builds.for_project(project_id)
        store.create(bid, env_id=env_id, suite_id=suite_id, name=name)
        store.set_total(bid, len(results))
        for r in results:
            store.record_result(bid, r)
        bad = sum(1 for r in results
                  if r["status"] in ("failed", "error", "timeout"))
        store.finish(bid, "failed" if bad else "passed")
        return bid


class TestTransitions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = _Fixture(self.tmp.name)
        self.env = self.fx.make_env("p1", "dev", latency_ms=10, fail_rate=0.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_fixed_persistent(self):
        base = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed"),
            _result("c2", "用例2", "failed"),
            _result("c3", "用例3", "failed"),
            _result("c4", "用例4", "passed"),
        ], env_id=self.env["id"], suite_id="s1")
        target = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed"),
            _result("c2", "用例2", "passed"),   # 已修复
            _result("c3", "用例3", "failed"),   # 持续失败
            _result("c4", "用例4", "failed"),   # 新增失败
        ], env_id=self.env["id"], suite_id="s1")

        d = self.fx.comparator.compare(base, target)
        t = d["transitions"]
        self.assertEqual([e["case_id"] for e in t["new_failures"]], ["c4"])
        self.assertEqual([e["case_id"] for e in t["fixed"]], ["c2"])
        self.assertEqual([e["case_id"] for e in t["persistent_failures"]], ["c3"])
        self.assertEqual(t["counts"]["still_passing"], 1)
        # 共同集通过率两边都是 50%，有进有退 → 持平
        self.assertEqual(d["summary"]["common"]["base_pass_rate"], 50.0)
        self.assertEqual(d["summary"]["common"]["target_pass_rate"], 50.0)
        self.assertEqual(d["verdict"]["code"], "similar")

    def test_verdict_improved_and_regressed(self):
        base = self.fx.make_build("p1", [
            _result("c1", "用例1", "failed"),
            _result("c2", "用例2", "failed"),
        ], env_id=self.env["id"])
        better = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed"),
            _result("c2", "用例2", "passed"),
        ], env_id=self.env["id"])
        d = self.fx.comparator.compare(base, better)
        self.assertEqual(d["verdict"]["code"], "improved")

        worse = self.fx.make_build("p1", [
            _result("c1", "用例1", "failed"),
            _result("c2", "用例2", "failed"),
        ], env_id=self.env["id"])
        d2 = self.fx.comparator.compare(better, worse)
        self.assertEqual(d2["verdict"]["code"], "regressed")

    def test_error_and_timeout_count_as_failure(self):
        base = self.fx.make_build("p1", [_result("c1", "用例1", "passed")],
                                  env_id=self.env["id"])
        target = self.fx.make_build("p1", [_result("c1", "用例1", "timeout")],
                                    env_id=self.env["id"])
        d = self.fx.comparator.compare(base, target)
        self.assertEqual(len(d["transitions"]["new_failures"]), 1)
        self.assertEqual(d["verdict"]["code"], "regressed")


class TestSetAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = _Fixture(self.tmp.name)
        self.env = self.fx.make_env("p1", "dev", latency_ms=10, fail_rate=0.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_partial_overlap(self):
        base = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed"),
            _result("c2", "用例2", "passed"),
            _result("c3", "用例3", "failed"),
        ], env_id=self.env["id"], suite_id="s1")
        target = self.fx.make_build("p1", [
            _result("c2", "用例2", "passed"),
            _result("c3", "用例3", "passed"),
            _result("c4", "用例4", "failed"),
        ], env_id=self.env["id"], suite_id="s2")

        d = self.fx.comparator.compare(base, target)
        c = d["comparability"]
        self.assertEqual(c["common_count"], 2)
        self.assertEqual(c["only_base_count"], 1)
        self.assertEqual(c["only_target_count"], 1)
        self.assertFalse(c["same_suite"])
        self.assertFalse(c["fully_comparable"])
        # 仅一边存在的用例被单独列出，不参与判定
        self.assertEqual([x["case_id"] for x in d["unmatched"]["only_base"]], ["c1"])
        self.assertEqual([x["case_id"] for x in d["unmatched"]["only_target"]], ["c4"])
        # c4 在对比构建里失败，但它不在共同集 → 不算「新增失败」
        self.assertEqual(d["transitions"]["counts"]["new_failure"], 0)
        # 共同集口径：c2/c3 从 1/2 通过变成 2/2 通过
        self.assertEqual(d["summary"]["common"]["base_pass_rate"], 50.0)
        self.assertEqual(d["summary"]["common"]["target_pass_rate"], 100.0)
        self.assertEqual(d["verdict"]["code"], "improved")
        # 套件不同要有说明
        self.assertTrue(any("不同套件" in n for n in c["notes"]))

    def test_no_common_cases_is_incomparable(self):
        base = self.fx.make_build("p1", [_result("c1", "用例1", "passed")],
                                  env_id=self.env["id"])
        target = self.fx.make_build("p1", [_result("c9", "用例9", "passed")],
                                    env_id=self.env["id"])
        d = self.fx.comparator.compare(base, target)
        self.assertEqual(d["verdict"]["code"], "incomparable")
        self.assertEqual(d["comparability"]["common_count"], 0)

    def test_same_name_different_id_not_paired(self):
        """同名但 id 不同的用例不得错配，要列入未配对并说明。"""
        base = self.fx.make_build("p1", [_result("c1", "登录接口", "passed")],
                                  env_id=self.env["id"])
        target = self.fx.make_build("p1", [_result("c9", "登录接口", "failed")],
                                    env_id=self.env["id"])
        d = self.fx.comparator.compare(base, target)
        self.assertEqual(d["comparability"]["common_count"], 0)
        collisions = d["unmatched"]["name_collisions"]
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["case_name"], "登录接口")
        self.assertTrue(any("同名" in n for n in d["comparability"]["notes"]))
        # 不错配 → 不会产生「新增失败」的误判
        self.assertEqual(d["transitions"]["counts"]["new_failure"], 0)

    def test_renamed_case_paired_by_id(self):
        base = self.fx.make_build("p1", [_result("c1", "旧名字", "passed")],
                                  env_id=self.env["id"])
        target = self.fx.make_build("p1", [_result("c1", "新名字", "passed")],
                                    env_id=self.env["id"])
        d = self.fx.comparator.compare(base, target)
        self.assertEqual(d["comparability"]["common_count"], 1)
        self.assertEqual(d["transitions"]["counts"]["renamed"], 1)
        self.assertTrue(any("改过名字" in n for n in d["comparability"]["notes"]))


class TestCrossEnvironment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = _Fixture(self.tmp.name)
        self.env_a = self.fx.make_env("p1", "dev", latency_ms=10, fail_rate=0.0)
        self.env_b = self.fx.make_env("p1", "staging", latency_ms=300, fail_rate=0.2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_env_mismatch_flagged_and_not_regression(self):
        base = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed", duration=0.1),
            _result("c2", "用例2", "passed", duration=0.2),
        ], env_id=self.env_a["id"])
        target = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed", duration=2.0),
            _result("c2", "用例2", "passed", duration=1.5),
        ], env_id=self.env_b["id"])

        d = self.fx.comparator.compare(base, target)
        c = d["comparability"]
        self.assertFalse(c["same_env"])
        self.assertTrue(any("环境不同" in n for n in c["notes"]))
        # 环境配置差异被列出
        keys = {x["key"] for x in c["env_diffs"]}
        self.assertIn("latency_ms", keys)
        self.assertIn("fail_rate", keys)
        # 跨环境：耗时差异只展示，不定性为「变慢」
        dc = d["duration_changes"]
        self.assertFalse(dc["same_env"])
        self.assertEqual(dc["slower"], [])
        self.assertEqual(len(dc["deltas"]), 2)
        self.assertIsNotNone(dc["note"])

    def test_same_env_duration_regression_detected(self):
        base = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed", duration=0.1),
            _result("c2", "用例2", "passed", duration=1.0),
        ], env_id=self.env_a["id"])
        target = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed", duration=0.12),  # 正常抖动
            _result("c2", "用例2", "passed", duration=3.0),   # 明显变慢
        ], env_id=self.env_a["id"])
        d = self.fx.comparator.compare(base, target)
        dc = d["duration_changes"]
        self.assertTrue(dc["same_env"])
        self.assertEqual([r["case_id"] for r in dc["slower"]], ["c2"])
        self.assertEqual(dc["faster"], [])

    def test_same_env_fully_comparable(self):
        results = [_result("c1", "用例1", "passed")]
        base = self.fx.make_build("p1", results, env_id=self.env_a["id"], suite_id="s1")
        target = self.fx.make_build("p1", results, env_id=self.env_a["id"], suite_id="s1")
        d = self.fx.comparator.compare(base, target)
        self.assertTrue(d["comparability"]["fully_comparable"])
        self.assertEqual(d["comparability"]["notes"], [])


class TestSummaryMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = _Fixture(self.tmp.name)
        self.env = self.fx.make_env("p1", "dev", latency_ms=10, fail_rate=0.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_overall_delta_and_distribution(self):
        base = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed", duration=0.05),
            _result("c2", "用例2", "failed", duration=2.0),
        ], env_id=self.env["id"])
        target = self.fx.make_build("p1", [
            _result("c1", "用例1", "passed", duration=0.06),
            _result("c2", "用例2", "passed", duration=0.6),
        ], env_id=self.env["id"])
        d = self.fx.comparator.compare(base, target)
        ov = d["summary"]["overall"]
        self.assertEqual(ov["base"]["pass_rate"], 50.0)
        self.assertEqual(ov["target"]["pass_rate"], 100.0)
        self.assertEqual(ov["delta"]["pass_rate"], 50.0)
        self.assertEqual(ov["delta"]["failed"], -1)
        dist = d["duration_distribution"]
        self.assertEqual(dist["base"], [1, 0, 0, 1, 0])
        self.assertEqual(dist["target"], [1, 0, 1, 0, 0])

    def test_coverage_compared_within_project(self):
        base = self.fx.make_build("p1", [_result("c1", "用例1", "passed")],
                                  env_id=self.env["id"])
        target = self.fx.make_build("p1", [_result("c1", "用例1", "passed")],
                                    env_id=self.env["id"])
        d = self.fx.comparator.compare(base, target)
        cov = d["summary"]["coverage"]
        self.assertTrue(cov["available"])
        self.assertIn("base", cov)
        self.assertIn("target", cov)
        self.assertIn("delta", cov)

    def test_coverage_skipped_across_projects(self):
        base = self.fx.make_build("p1", [_result("c1", "用例1", "passed")],
                                  env_id=self.env["id"])
        target = self.fx.make_build("p2", [_result("c1", "用例1", "passed")])
        d = self.fx.comparator.compare(base, target)
        self.assertFalse(d["summary"]["coverage"]["available"])
        self.assertFalse(d["comparability"]["same_project"])
        self.assertTrue(any("不同项目" in n for n in d["comparability"]["notes"]))

    def test_errors(self):
        bid = self.fx.make_build("p1", [_result("c1", "用例1", "passed")],
                                 env_id=self.env["id"])
        self.assertIn("error", self.fx.comparator.compare(bid, bid))
        self.assertIn("error", self.fx.comparator.compare(bid, "build_missing"))
        self.assertIn("error", self.fx.comparator.compare("build_missing", bid))


class TestCompareApi(unittest.TestCase):
    """API 路由接线：GET /api/builds/compare?base=..&target=.."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = _Fixture(self.tmp.name)
        try:
            from flask import Flask
            from web import api
        except ImportError:
            self.skipTest("Flask 不可用")
        app = Flask(__name__)
        app.config["STORE_REGISTRY"] = self.fx.registry
        app.config["BUILD_REGISTRY"] = self.fx.builds
        app.config["COMPARE"] = self.fx.comparator
        app.register_blueprint(api)
        self.client = app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_compare_endpoint(self):
        env = self.fx.make_env("p1", "dev", latency_ms=10, fail_rate=0.0)
        base = self.fx.make_build("p1", [_result("c1", "用例1", "failed")],
                                  env_id=env["id"])
        target = self.fx.make_build("p1", [_result("c1", "用例1", "passed")],
                                    env_id=env["id"])
        resp = self.client.get(f"/api/builds/compare?base={base}&target={target}")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["verdict"]["code"], "improved")
        self.assertEqual(data["transitions"]["counts"]["fixed"], 1)

    def test_compare_endpoint_validation(self):
        resp = self.client.get("/api/builds/compare")
        self.assertEqual(resp.status_code, 400)
        resp = self.client.get("/api/builds/compare?base=x&target=y")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
