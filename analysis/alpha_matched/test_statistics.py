"""Invented panels only. Run from code with -m unittest analysis.alpha_matched.test_statistics."""
import unittest

import numpy as np
import pandas as pd

from .statistics import (ARMS, HEADS, cohort_report, equal_cohort_auc,
                         participant_probabilities, primary_contrasts, select_inner_oof_threshold)


BINARY = ("non-IBD", "IBD")
THREE = ("non-IBD", "UC", "CD")


def roster(labels, persons=None, groups=None):
    n = len(labels)
    people = persons if persons is not None else [f"p{i}" for i in range(n)]
    return pd.DataFrame(dict(person=people, group=people if groups is None else groups,
                             y=labels, cohort=["fixture"] * n),
                        index=[f"v{i}" for i in range(n)])


def binary(meta, scores):
    p = np.asarray(scores)
    return pd.DataFrame(np.column_stack((1 - p, p)), index=meta.index, columns=BINARY)


def all_models(pred):
    models = {(a, h): pred for a in ARMS for h in HEADS}
    rule = .5 if pred.shape[1] == 2 else None
    return models, dict.fromkeys(models, rule)


def single(meta, pred, threshold=.5):
    return cohort_report(meta, {"model": pred}, tuple(pred.columns), {"model": threshold})["model"]


class StatisticsTests(unittest.TestCase):
    def test_hand_scored_longitudinal_ties_and_calibration(self):
        m = roster([1, 1, 0, 1, 0], ["a", "a", "b", "c", "d"])
        pred = binary(m, [.8, 1., .6, .6, .2])
        means = participant_probabilities(m, pred.iloc[::-1], BINARY)
        self.assertAlmostEqual(means.loc["a", "IBD"], .9)
        entry = single(m, pred.iloc[::-1])
        self.assertEqual(entry["coverage"]["visits_per_person"], {1: 3, 2: 1})
        self.assertEqual(entry["coverage"]["eligible_people"], 4)
        s = entry["metrics"]
        for key, expected in dict(roc_auc=7/8, pr_auc=5/6, balanced_accuracy=3/4,
                                  macro_f1=(4/5 + 2/3)/2, brier=.1425).items():
            self.assertAlmostEqual(s[key], expected)
        bin6 = s["by_class"]["IBD"]["calibration"][6]
        self.assertEqual(bin6, dict(bin=6, people=2, predicted=.6, observed=.5))

    def test_three_way_and_fixed_axis_with_absent_subtype(self):
        m = roster([0, 1, 2])
        p = pd.DataFrame([[.6, .2, .2], [.1, .3, .6], [.1, .2, .7]],
                         columns=THREE, index=m.index)
        s = single(m, p, None)["metrics"]
        self.assertEqual(s["roc_auc"], 1)
        self.assertEqual(s["pr_auc"], 1)
        self.assertAlmostEqual(s["brier"], (.24 + .86 + .14)/3)
        self.assertAlmostEqual(s["macro_f1"], 5/9)
        self.assertAlmostEqual(s["balanced_accuracy"], 2/3)
        reduced = single(m.iloc[:2], p.iloc[:2], None)["metrics"]
        self.assertIsNone(reduced["roc_auc"])
        self.assertIsNone(reduced["by_class"]["CD"]["recall"])
        self.assertEqual(reduced["by_class"]["UC"]["recall"], 0)
        models, rules = all_models(p)
        models["RAW", "Ricci"] = pd.DataFrame(np.full((3, 3), 1/3), columns=THREE, index=m.index)
        out = primary_contrasts(m, models, THREE, rules, draws=40)
        self.assertAlmostEqual(out[("RAW", "joint"), ("RAW", "Ricci")]["metrics"]["roc_auc"]["difference"], .5)

    def test_one_class_control_only_and_empty(self):
        m = roster([0, 0, 0])
        p = binary(m, [.2, .4, .8])
        s = single(m, p)["metrics"]
        for key in ("roc_auc", "pr_auc", "macro_f1", "balanced_accuracy"):
            self.assertIsNone(s[key])
        self.assertAlmostEqual(s["by_class"]["non-IBD"]["recall"], 2/3)
        self.assertAlmostEqual(s["brier"], .84/3)
        models, rules = all_models(p)
        rules["RAW", "Ricci"] = .9
        contrasts = primary_contrasts(m, models, BINARY, rules, draws=20)
        r = contrasts[("RAW", "joint"), ("RAW", "Ricci")]["metrics"]["recall/non-IBD"]
        self.assertAlmostEqual(r["difference"], -1/3)
        self.assertEqual(r["valid_draws"], 20)
        self.assertEqual(single(m.iloc[:0], p.iloc[:0])["status"], "no eligible visits")

    def test_missing_visit_suppresses_whole_cohort_comparison(self):
        m = roster([1, 1, 0], ["a", "a", "b"])
        p = binary(m, [.3, .6, .2])
        models, rules = all_models(p)
        models["RAW", "joint"] = p.iloc[:1]
        reports = cohort_report(m, models, BINARY, rules)
        entry = reports["RAW", "joint"]
        self.assertEqual(entry["status"], "incomplete")
        self.assertIsNone(entry["metrics"])
        self.assertEqual(entry["coverage"]["partial_people"], 1)
        self.assertEqual(entry["coverage"]["missing_people"], 1)
        pairs = primary_contrasts(m, models, BINARY, rules, draws=10)
        for pair, info in pairs.items():
            if ("RAW", "joint") in pair:
                self.assertEqual(info["metrics"], {})
                self.assertEqual(info["status"], "incomplete")
        self.assertIsNone(equal_cohort_auc({"A": reports}, ("RAW", "joint"), expected_cohorts=["A"]))

    def test_paired_family_draws_against_literal_replication(self):
        # Mixed-label related participants always resample together.
        m = roster([0, 1] * 4, groups=list("aabbccdd"))
        x = np.array([.2, .4, .4, .2, .6, .8, .9, .9])
        z = np.array([.3, .9, .4, .6, .9, .8, .4, .5])
        models, rules = all_models(binary(m, z))
        models["RAW", "joint"] = binary(m, x)
        pairs = primary_contrasts(m, models, BINARY, rules, draws=150, seed=29)
        self.assertEqual(len(pairs), 7)

        def auc(indices, scores):
            pos = scores[[i for i in indices if m.y.iloc[i] == 1]]
            neg = scores[[i for i in indices if m.y.iloc[i] == 0]]
            return sum(float(a > b) + .5 * (a == b) for a in pos for b in neg) / (len(pos)*len(neg))

        rng = np.random.default_rng(29)
        diffs = []
        for _ in range(150):
            indices = [i for family in rng.integers(4, size=4) for i in [2*family, 2*family+1]]
            diffs.append(auc(indices, x) - auc(indices, z))
        for pair in [(("RAW", "joint"), ("RELATIVE", "joint")), (("RAW", "joint"), ("RAW", "H0"))]:
            metric = pairs[pair]["metrics"]["roc_auc"]
            np.testing.assert_allclose(metric["interval"], np.quantile(diffs, [.025, .975]))
            self.assertEqual(metric["valid_draws"], 150)
            self.assertAlmostEqual(metric["difference"], auc(range(8), x) - auc(range(8), z))
        self.assertEqual(pairs[("RELATIVE", "joint"), ("RELATIVE", "Ricci")]["metrics"]["roc_auc"]["interval"], [0, 0])

    def test_sparse_bootstrap_and_equal_cohort_weight(self):
        m = roster([0, 1])
        p = binary(m, [.7, .1])
        models, rules = all_models(p)
        pairs = primary_contrasts(m, models, BINARY, rules, draws=100)
        item = next(iter(pairs.values()))["metrics"]["roc_auc"]
        self.assertTrue(0 < item["valid_draws"] < 90)
        self.assertIsNone(item["interval"])
        big = roster([0, 1]*5)
        reports = {"A": {"model": single(m, p)}, "B": {"model": single(big, binary(big, [.1, .8]*5))},
                   "C": {"model": single(m.iloc[:1], p.iloc[:1])}}
        self.assertEqual(equal_cohort_auc(reports, "model", expected_cohorts="ABC"), .5)
        with self.assertRaises(ValueError):
            equal_cohort_auc(reports, "model", expected_cohorts="ABCD")

    def test_threshold_uses_complete_parent_oof_and_family_closure(self):
        m = roster([0, 0, 1, 0, 1], list("aabcd"), list("xxxyy"))
        p = binary(m, [.51, .69, .75, .73, .8])
        folds = [({"y"}, p.iloc[:3]), ({"x"}, p.iloc[3:])]
        self.assertEqual(select_inner_oof_threshold(m, BINARY, folds), .75)
        for bad in [[({"x", "y"}, p.iloc[:3]), folds[1]],
                    [({"y"}, p.iloc[:2]), folds[1]], folds[:1], folds + folds]:
            with self.assertRaises(ValueError):
                select_inner_oof_threshold(m, BINARY, bad)
        # All tied scores, equal class recalls on average for any threshold.
        q = binary(m, [.4]*5)
        self.assertEqual(select_inner_oof_threshold(m, BINARY, [({"y"}, q.iloc[:3]), ({"x"}, q.iloc[3:])]), .5)

    def test_axes_probabilities_manifest_integrity(self):
        m = roster([0, 1])
        p = binary(m, [.2, .8])
        for invalid in [p.iloc[:, ::-1], p * 0, p * np.nan, p * 2,
                        p.rename(index={"v1": "unlisted"}), pd.concat([p, p])]:
            with self.assertRaises(ValueError):
                cohort_report(m, {"m": invalid}, BINARY, {"m": .5})
        for invalid_meta in [m.assign(person="same"), m.assign(y=3), m.assign(group=None),
                             m.assign(cohort=["one", "two"])]:
            with self.assertRaises(ValueError):
                single(invalid_meta, p)


if __name__ == "__main__":
    unittest.main()
