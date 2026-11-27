"""평가 지표와 모델 선택의 회귀 검사. unittest만 사용하며 PPO 학습은 실행하지 않는다."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common_v5 as V
import compare_v5 as compare

# ppo_v5의 명령줄 상대 경로 설정이 테스트 실행 위치를 바꾸지 않도록 복구한다.
_cwd = os.getcwd()
import ppo_v5 as ppo
os.chdir(_cwd)


class ZeroPolicy:
    def reset(self, n):
        self.n = n

    def act(self, states):
        return np.zeros(len(states))


def scripted_run(states):
    """알려진 상태 이력으로 카운터를 검사한다. states: (500,N,4)."""
    with patch.object(V, "step_batch", side_effect=[s.copy() for s in states]):
        return V.batch_run(ZeroPolicy(), np.zeros_like(states[0]), record=True)


class UprightDurationTests(unittest.TestCase):
    def test_full_ten_seconds_and_no_initial_extra_sample(self):
        result = V.batch_run(ZeroPolicy(), np.zeros((1, 4)), record=True)
        self.assertEqual(result["upright_time_s"][0], V.C.SIM_TIME)
        self.assertEqual(result["upright_fraction"][0], 1.0)
        self.assertEqual(len(result["S_hist"]), V.N_STEPS + 1)
        self.assertNotIn("success", result)

    def test_disjoint_intervals_are_added_and_final_window_is_irrelevant(self):
        states = np.zeros((V.N_STEPS, 1, 4))
        upright = np.arange(V.N_STEPS) % 4 < 2
        states[~upright, 0, 2] = np.deg2rad(4.0)
        result = scripted_run(states)
        self.assertAlmostEqual(result["upright_time_s"][0], 5.0)
        self.assertFalse(result["final_upright"][0])
        self.assertTrue(np.isnan(result["settling_time_s"][0]))

    def test_late_recovery_is_counted_without_two_second_requirement(self):
        states = np.zeros((V.N_STEPS, 1, 4))
        states[:450, 0, 2] = np.deg2rad(4.0)
        result = scripted_run(states)
        self.assertAlmostEqual(result["upright_time_s"][0], 1.0)
        self.assertTrue(result["final_upright"][0])
        self.assertAlmostEqual(result["settling_time_s"][0], 9.02)

    def test_angular_velocity_required_but_cart_speed_is_secondary(self):
        states = np.zeros((V.N_STEPS, 2, 4))
        states[:, 0, 1] = 0.1
        states[:, 1, 3] = V.CRIT["theta_dot"] + 0.01
        result = scripted_run(states)
        np.testing.assert_allclose(result["upright_time_s"], [10.0, 0.0])
        np.testing.assert_allclose(result["stationary_upright_time_s"], [0.0, 0.0])

    def test_limits_are_inclusive(self):
        states = np.array([[0, 0, np.deg2rad(3.0), 0.2],
                           [0, 0, -np.deg2rad(3.0), -0.2],
                           [0, 0, np.deg2rad(3.01), 0.0]])
        np.testing.assert_array_equal(V.stable_mask(states, True), [True, True, False])

    def test_rail_exit_ends_credit_even_if_frozen_state_is_upright(self):
        states = np.zeros((V.N_STEPS, 1, 4))
        states[9:, 0, 0] = V.C.X_MAX + 0.01
        result = scripted_run(states)
        self.assertAlmostEqual(result["upright_time_s"][0], 9 * V.C.DT)
        self.assertAlmostEqual(result["observed_time_s"][0], 10 * V.C.DT)
        self.assertTrue(result["failed"][0])
        self.assertEqual(np.count_nonzero(result["alive_hist"][1:, 0]), 9)
        stats = compare.response_stats("test", result["S_hist"][:, 0], result["F_hist"][:, 0])
        self.assertAlmostEqual(stats["upright_time_s"], result["upright_time_s"][0])
        self.assertTrue(np.isnan(stats["ts_s"]))

    def test_initial_rail_exit_gets_no_credit(self):
        result = V.batch_run(ZeroPolicy(), np.array([[V.C.X_MAX + 0.01, 0, 0, 0]]))
        self.assertEqual(result["upright_time_s"][0], 0.0)
        self.assertEqual(result["observed_time_s"][0], 0.0)
        self.assertTrue(result["failed"][0])

    def test_summary_includes_rail_failures(self):
        states = np.zeros((V.N_STEPS, 2, 4))
        states[9:, 0, 0] = V.C.X_MAX + 0.01
        result = scripted_run(states)
        summary = V.summarize(result)
        self.assertAlmostEqual(summary["upright_time_s"], (9 * V.C.DT + 10.0) / 2)
        self.assertEqual(summary["fail_rail"], 1)
        self.assertEqual(summary["n_survived"], 1)
        self.assertNotIn("success_rate", summary)
        self.assertEqual(summary["evaluation_schema"], V.EVALUATION_SCHEMA)

    def test_trial_exports_and_paired_differences_use_all_trials(self):
        states = np.zeros((V.N_STEPS, 2, 4))
        states[9:, 0, 0] = V.C.X_MAX + 0.01
        a = scripted_run(states)
        b = V.batch_run(ZeroPolicy(), np.zeros((2, 4)))
        meta = [dict(theta0_deg=5, trial=i) for i in range(2)]
        exported = compare.trial_rows({"PID": a}, np.zeros((2, 4)), meta)
        self.assertIn("upright_time_s", exported[0])
        self.assertNotIn("success", exported[0])
        rows = compare.paired({"PID": a, "LQR": b}, meta, "PID", "LQR")
        overall = next(r for r in rows if r["theta0_abs_deg"] == "전체")
        self.assertEqual(overall["upright_n"], 2)
        self.assertLess(overall["upright_mean"], 0)


def selection_score(upright_time, survived=148, mean_return=100):
    return dict(evaluation_schema=V.EVALUATION_SCHEMA, upright_time_s=upright_time,
                upright_fraction=upright_time / 10, n_survived=survived, n=148,
                fail_rail=148 - survived, rmse_deg=3.0, effort=40.0, mean_return=mean_return)


class PPOSelectionTests(unittest.TestCase):
    def test_duration_takes_priority_over_old_success_or_reward(self):
        self.assertGreater(ppo.key(selection_score(9.0, 147, 10)),
                           ppo.key(selection_score(8.0, 148, 500)))
        self.assertGreater(ppo.key(selection_score(9.0, 148, 10)),
                           ppo.key(selection_score(9.0, 147, 500)))

    def test_old_history_is_not_silently_reused(self):
        with self.assertRaisesRegex(ValueError, "legacy"):
            ppo.key(dict(n_success=148, mean_return=500))

    def test_legacy_checkpoints_are_rescored_and_originals_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "models/ppo_v5_runs/seed0"
            (source / "ckpt").mkdir(parents=True)
            history = [dict(step=100, n_success=148), dict(step=200, n_success=0)]
            original = json.dumps(history)
            (source / "history.json").write_text(original)
            for step in (100, 200):
                (source / "ckpt" / f"{step:08d}.zip").write_bytes(str(step).encode())
            fake_ppo = types.SimpleNamespace(load=lambda path, device: path)
            def score(policy, states):
                step = int(Path(policy.model).stem)
                return None, selection_score(8.0 if step == 100 else 9.0)
            cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.multiple(ppo, RUN_DIR="models/ppo_v5_upright_runs", BEST_PATH="models/ppo_v5_upright",
                                    SUMMARY_PATH="models/ppo_v5_upright.json", TAG=""), \
                     patch.dict(sys.modules, {"stable_baselines3": types.SimpleNamespace(PPO=fake_ppo)}), \
                     patch.object(ppo, "evaluate_policy", side_effect=score), \
                     contextlib.redirect_stdout(io.StringIO()):
                    ppo.select(legacy=True)
                self.assertEqual((root / "models/ppo_v5_upright.zip").read_bytes(), b"200")
                self.assertEqual((source / "history.json").read_text(), original)
                chosen = json.loads((root / "models/ppo_v5_upright.json").read_text())
                self.assertEqual(chosen["best"]["selected_step"], 200)
                self.assertNotIn("n_success", chosen["best"])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
