"""构建对比（engine.compare）单元测试。

覆盖：用例集合对齐（case_id 主键 / 唯一名字兜底 / 同名不同 id 不合并）、
结果迁移分类（新增失败 / 已修复 / 持续失败 / 其他变化）、双口径通过率、
耗时退化检测、跨环境置信度与注意事项、结论判定、对比 API。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import BuildComparator, CoverageAnalyzer, EnvironmentManager
from engine.compare import _align
from storage import BuildStoreRegistry, StoreRegistry


def R(cid, name, status, duration=0.1, group="g", priority="P2", **kw):
    """构造一条用例结果。"""
    r = {
        "case_id": cid,
        "case_name": name,
        "group": group,
        "priority": priority,
        "status": status,
        "duration": duration,
        "steps": [],
        "assertions": [],
        "logs": [],
        "message": "",
    }
    r.update(kw)
    if status in ("failed", "error", "timeout") and not r["assertions"]:
        r["assertions"] = [{"ok": False, "message": f"{name} 断言失败"}]
    return r


class CompareFixture(unittest.TestCase):
    """搭好存储 / 环境 / 覆盖率与对比器的基类。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.builds = BuildStoreRegistry(os.path.join(self.tmp.name, "builds"))
        self.stores = StoreRegistry(os.path.join(self.tmp.name, "store"))
        self.env_mgr = EnvironmentManager(self.stores, self.tmp.name)
        self.coverage = CoverageAnalyzer(self.builds)
        self.comparator = BuildComparator(self.builds, self.stores,
                                          self.env_mgr, self.coverage)

    def tearDown(self):
        self.tmp.cleanup()

    def make_env(self, project_id, name, latency=20, fail_rate=0.0):
        return self.env_mgr.create(project_id, {
            "name": name,
            "config": {"base_url": "http://mock.local",
                       "latency_ms": latency, "fail_rate": fail_rate},
        })

    def make_build(self, project_id, build_id, results,
                   env_id="e1", suite_id="s1"):
        store = self.builds.for_project(project_id)
        store.create(build_id, env_id=env_id, suite_id=suite_id, name=build_id)
        store.set_total(build_id, len(results))
        for r in results:
            store.record_result(build_id, r)
        failed = sum(1 for r in results
                     if r["status"] in ("failed", "error", "timeout"))
        store.finish(build_id, "failed" if failed else "passed")
        return build_id


class TestAlign(unittest.TestCase):
    """用例集合对齐：不丢单侧用例、不错配同名用例。"""

    def test_match_by_case_id(self):
        matched, only_b, only_t, amb = _align(
            [R("c1", "登录", "passed"), R("c2", "下单", "failed")],
            [R("c1", "登录", "passed"), R("c2", "下单", "passed")],
        )
        self.assertEqual({k for k, _, _ in matched}, {"c1", "c2"})
        self.assertEqual(only_b, [])
        self.assertEqual(only_t, [])

    def test_only_one_side_kept_not_dropped(self):
        matched, only_b, only_t, amb = _align(
            [R("c1", "登录", "passed"), R("c2", "旧用例", "passed")],
            [R("c1", "登录", "passed"), R("c3", "新用例", "failed")],
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual([r["case_id"] for r in only_b], ["c2"])
        self.assertEqual([r["case_id"] for r in only_t], ["c3"])

    def test_same_name_different_id_not_merged(self):
        """同名但 id 不同（删除重建）→ 不合并，进 ambiguous。"""
        matched, only_b, only_t, amb = _align(
            [R("c1", "登录", "failed")],
            [R("c9", "登录", "passed")],
        )
        self.assertEqual(matched, [])
        self.assertEqual(len(amb), 1)
        self.assertEqual(amb[0]["name"], "登录")
        self.assertEqual(amb[0]["base_case_ids"], ["c1"])
        self.assertEqual(amb[0]["target_case_ids"], ["c9"])
        # 同时保留在单侧清单里，不丢数据
        self.assertEqual(len(only_b), 1)
        self.assertEqual(len(only_t), 1)

    def test_missing_id_falls_back_to_unique_name(self):
        matched, only_b, only_t, amb = _align(
            [R(None, "健康检查", "passed")],
            [R(None, "健康检查", "failed")],
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][0], "name:健康检查")

    def test_missing_id_duplicate_name_not_matched(self):
        """缺 id 且名字在单侧重复 → 放弃对齐，避免错配。"""
        matched, only_b, only_t, amb = _align(
            [R(None, "重名", "passed"), R(None, "重名", "failed")],
            [R(None, "重名", "passed")],
        )
        self.assertEqual(matched, [])
        self.assertEqual(len(only_b), 2)
        self.assertEqual(len(only_t), 1)


class TestClassification(CompareFixture):
    def test_new_failure_fixed_still_failing(self):
        self.make_build("p1", "b1", [
            R("c1", "稳定通过", "passed"),
            R("c2", "被修复", "failed"),
            R("c3", "老问题", "failed"),
            R("c4", "新退化", "passed"),
        ])
        self.make_build("p1", "b2", [
            R("c1", "稳定通过", "passed"),
            R("c2", "被修复", "passed"),
            R("c3", "老问题", "error"),
            R("c4", "新退化", "failed"),
        ])
        r = self.comparator.compare("b1", "b2")
        cases = r["cases"]
        self.assertEqual([x["case_id"] for x in cases["new_failures"]], ["c4"])
        self.assertEqual([x["case_id"] for x in cases["fixed"]], ["c2"])
        self.assertEqual([x["case_id"] for x in cases["still_failing"]], ["c3"])
        self.assertEqual(cases["stable_count"], 1)
        # 失败原因从目标侧提取
        self.assertEqual(cases["new_failures"][0]["reason"], "新退化 断言失败")
        # 有新增失败且无修复 → 退化
        self.assertEqual(r["verdict"]["level"], "mixed")  # 既有新增失败又有修复
        self.assertEqual(r["summary"]["matched"]["count"], 4)

    def test_verdict_levels(self):
        # 只有新增失败 → regression
        self.make_build("p1", "b1", [R("c1", "a", "passed")])
        self.make_build("p1", "b2", [R("c1", "a", "failed")])
        self.assertEqual(self.comparator.compare("b1", "b2")["verdict"]["level"],
                         "regression")
        # 只有修复 → improved
        self.make_build("p1", "b3", [R("c1", "a", "failed")])
        self.make_build("p1", "b4", [R("c1", "a", "passed")])
        self.assertEqual(self.comparator.compare("b3", "b4")["verdict"]["level"],
                         "improved")
        # 无变化 → stable
        self.make_build("p1", "b5", [R("c1", "a", "passed")])
        self.make_build("p1", "b6", [R("c1", "a", "passed")])
        self.assertEqual(self.comparator.compare("b5", "b6")["verdict"]["level"],
                         "stable")

    def test_inconclusive_when_no_overlap(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")])
        self.make_build("p1", "b2", [R("c2", "b", "passed")])
        r = self.comparator.compare("b1", "b2")
        self.assertEqual(r["verdict"]["level"], "inconclusive")
        self.assertEqual(r["summary"]["matched"]["count"], 0)
        # 单侧用例仍然列出，不被吞掉
        self.assertEqual(len(r["cases"]["only_base"]), 1)
        self.assertEqual(len(r["cases"]["only_target"]), 1)

    def test_skipped_transition_is_other_change_not_new_failure(self):
        """skipped → failed 不算「新增失败」（基准时并未通过）。"""
        self.make_build("p1", "b1", [R("c1", "a", "skipped", 0.0)])
        self.make_build("p1", "b2", [R("c1", "a", "failed")])
        r = self.comparator.compare("b1", "b2")
        self.assertEqual(r["cases"]["new_failures"], [])
        self.assertEqual(len(r["cases"]["other_changes"]), 1)
        self.assertEqual(r["cases"]["other_changes"][0]["base_status"], "skipped")


class TestPassRates(CompareFixture):
    def test_matched_pass_rate_is_apples_to_apples(self):
        """整体通过率受集合差异影响，共同用例口径才严格可比。"""
        # 基准：c1 过、c2 挂 → 50%；目标：c1 过、c2 不在、c3 过 → 100%
        self.make_build("p1", "b1", [R("c1", "a", "passed"), R("c2", "b", "failed")])
        self.make_build("p1", "b2", [R("c1", "a", "passed"), R("c3", "c", "passed")])
        r = self.comparator.compare("b1", "b2")
        self.assertEqual(r["summary"]["base"]["pass_rate"], 50.0)
        self.assertEqual(r["summary"]["target"]["pass_rate"], 100.0)
        # 共同用例只有 c1：两边都是 100%，真实变化为 0
        m = r["summary"]["matched"]
        self.assertEqual(m["count"], 1)
        self.assertEqual(m["base_pass_rate"], 100.0)
        self.assertEqual(m["target_pass_rate"], 100.0)
        self.assertEqual(m["pass_rate_delta"], 0.0)
        # 注意事项里说明集合差异
        self.assertTrue(any("仅基准存在 1 条" in c for c in r["caveats"]))


class TestDurations(CompareFixture):
    def test_duration_regression_detected(self):
        self.make_build("p1", "b1", [
            R("c1", "稳定", "passed", 0.10),
            R("c2", "变慢", "passed", 0.20),
            R("c3", "抖动", "passed", 0.10),
        ])
        self.make_build("p1", "b2", [
            R("c1", "稳定", "passed", 0.10),
            R("c2", "变慢", "passed", 1.20),   # +1.0s / +500% → 显著变慢
            R("c3", "抖动", "passed", 0.12),   # 毫秒级抖动 → 不算
        ])
        r = self.comparator.compare("b1", "b2")
        slower = r["durations"]["slower"]
        self.assertEqual([x["case_id"] for x in slower], ["c2"])
        self.assertAlmostEqual(slower[0]["duration_delta"], 1.0, places=2)
        self.assertEqual(slower[0]["duration_pct"], 500.0)
        self.assertFalse(r["durations"]["env_influenced"])
        # 分桶覆盖全部用例
        total = sum(b["base"] for b in r["durations"]["buckets"])
        self.assertEqual(total, 3)

    def test_skipped_case_excluded_from_duration_changes(self):
        self.make_build("p1", "b1", [R("c1", "a", "skipped", 0.0)])
        self.make_build("p1", "b2", [R("c1", "a", "passed", 3.0)])
        r = self.comparator.compare("b1", "b2")
        self.assertEqual(r["durations"]["slower"], [])


class TestCrossEnvironment(CompareFixture):
    def test_cross_env_lowers_confidence_and_explains(self):
        env_a = self.make_env("p1", "dev 环境", latency=15, fail_rate=0.0)
        env_b = self.make_env("p1", "staging 环境", latency=200, fail_rate=0.15)
        self.make_build("p1", "b1", [R("c1", "a", "passed", 0.1)],
                        env_id=env_a["id"], suite_id="s1")
        self.make_build("p1", "b2", [R("c1", "a", "failed", 0.8)],
                        env_id=env_b["id"], suite_id="s1")
        r = self.comparator.compare("b1", "b2")
        # 置信度降级 + 环境差异提示
        self.assertEqual(r["confidence"], "low")
        self.assertFalse(r["same_env"])
        self.assertTrue(any("运行环境不同" in c for c in r["caveats"]))
        self.assertTrue(any("staging" in c for c in r["caveats"]))
        # 环境配置差异表能定位到具体配置项
        keys = {i["key"] for i in r["env_diff"]["items"]}
        self.assertIn("config.latency_ms", keys)
        self.assertIn("config.fail_rate", keys)
        lat = next(i for i in r["env_diff"]["items"]
                   if i["key"] == "config.latency_ms")
        self.assertEqual((lat["base"], lat["target"]), (15, 200))
        # 跨环境时耗时对比标记为「受环境影响」
        self.assertTrue(r["durations"]["env_influenced"])

    def test_same_env_high_confidence(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")],
                        env_id="e1", suite_id="s1")
        self.make_build("p1", "b2", [R("c1", "a", "passed")],
                        env_id="e1", suite_id="s1")
        r = self.comparator.compare("b1", "b2")
        self.assertEqual(r["confidence"], "high")
        self.assertEqual(r["caveats"], [])

    def test_cross_suite_medium_confidence(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")],
                        env_id="e1", suite_id="s1")
        self.make_build("p1", "b2", [R("c1", "a", "passed")],
                        env_id="e1", suite_id="s2")
        r = self.comparator.compare("b1", "b2")
        self.assertEqual(r["confidence"], "medium")
        self.assertTrue(any("不同套件" in c for c in r["caveats"]))


class TestCoverageAndLinks(CompareFixture):
    def test_coverage_delta_and_file_changes(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")])
        self.make_build("p1", "b2", [R("c1", "a", "passed")])
        r = self.comparator.compare("b1", "b2")
        cov = r["coverage"]
        self.assertIsNotNone(cov)
        self.assertIn("percent", cov["base"])
        self.assertAlmostEqual(
            cov["delta"],
            round(cov["target"]["percent"] - cov["base"]["percent"], 1), places=3)
        self.assertLessEqual(len(cov["file_changes"]), 10)

    def test_links_point_to_build_details(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")])
        self.make_build("p1", "b2", [R("c1", "a", "passed")])
        r = self.comparator.compare("b1", "b2")
        self.assertIn("project_id=p1", r["links"]["base"]["report"])
        self.assertIn("build_id=b1", r["links"]["base"]["report"])
        self.assertIn("build_id=b2", r["links"]["target"]["report"])

    def test_missing_build_returns_error(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")])
        self.assertIn("error", self.comparator.compare("b1", "nope"))
        self.assertIn("error", self.comparator.compare("nope", "b1"))


class TestCompareApi(CompareFixture):
    """对比接口：参数校验与正常返回。"""

    def _app(self):
        from flask import Flask
        from web import api
        app = Flask(__name__)
        app.config["BUILD_REGISTRY"] = self.builds
        app.config["STORE_REGISTRY"] = self.stores
        app.config["ENV_MANAGER"] = self.env_mgr
        app.config["COVERAGE"] = self.coverage
        app.config["COMPARATOR"] = self.comparator
        app.register_blueprint(api)
        return app.test_client()

    def setUp(self):
        super().setUp()
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("Flask 未安装，跳过 API 层测试")

    def test_compare_endpoint(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")])
        self.make_build("p1", "b2", [R("c1", "a", "failed")])
        client = self._app()
        resp = client.get("/api/builds/compare?base=b1&target=b2")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["verdict"]["level"], "regression")
        self.assertEqual(data["cases"]["new_failures"][0]["case_id"], "c1")

    def test_compare_endpoint_validates_params(self):
        self.make_build("p1", "b1", [R("c1", "a", "passed")])
        client = self._app()
        self.assertEqual(client.get("/api/builds/compare").status_code, 400)
        self.assertEqual(client.get("/api/builds/compare?base=b1&target=b1")
                         .status_code, 400)
        self.assertEqual(client.get("/api/builds/compare?base=b1&target=nope")
                         .status_code, 404)


if __name__ == "__main__":
    unittest.main()
