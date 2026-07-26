import unittest

from app.planner import build_query_plan


class PlannerTests(unittest.TestCase):
    def test_direct_path_for_existing_multi_metric_question(self) -> None:
        plan = build_query_plan("What was India GDP and CO2 emission in 2022 and explain their impact?")
        self.assertEqual(plan.strategy, "direct")
        self.assertEqual(len(plan.steps), 1)

    def test_decomposes_compare_question(self) -> None:
        plan = build_query_plan("Compare India and China GDP in 2022")
        self.assertEqual(plan.strategy, "decomposed")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].subquestion, "What was India GDP in 2022?")
        self.assertEqual(plan.steps[1].subquestion, "What was China GDP in 2022?")


if __name__ == "__main__":
    unittest.main()
