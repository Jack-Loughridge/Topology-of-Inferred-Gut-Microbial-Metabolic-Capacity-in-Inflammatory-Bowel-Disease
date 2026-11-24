"""Participant-level statistics on a frozen endpoint's eligible visit roster.

Roster index = globally unique visit ID, columns = person, group, cohort, y.
y is an integer index into the ordered class names on every probability table.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ARMS = ("RAW", "RELATIVE")
HEADS = ("H0", "Ricci", "joint")
PRIMARY_PAIRS = tuple(
    ((arm, "joint"), (arm, single)) for arm in ARMS for single in HEADS[:2]
) + tuple((("RAW", head), ("RELATIVE", head)) for head in HEADS)


def _check_roster(roster, classes):
    if (len(classes) < 2 or len(set(classes)) != len(classes)
            or any(not isinstance(c, str) or not c.strip() for c in classes)):
        raise ValueError("Specify unique ordered class names")
    keys = ["cohort", "person", "group", "y"]
    if (not roster.index.is_unique or roster.index.hasnans
            or roster[keys].isna().any().any()):
        raise ValueError("Missing metadata or non-unique visit IDs")
    if len(roster) and (roster.y.dtype.kind not in "iu"
                        or not roster.y.between(0, len(classes) - 1).all()):
        raise ValueError("Expected integer endpoint labels in class order")
    if (roster.groupby("person")[["cohort", "group", "y"]].nunique() > 1).any().any():
        raise ValueError("Inconsistent participant metadata")


@dataclass
class _Panel:
    people: pd.DataFrame
    probabilities: pd.DataFrame
    coverage: dict

    @property
    def complete(self):
        return self.coverage["eligible_visits"] == self.coverage["predicted_visits"]


def _aggregate(roster, probabilities, classes):
    if (list(probabilities.columns) != list(classes) or not probabilities.index.is_unique
            or not probabilities.index.isin(roster.index).all()):
        raise ValueError("Invalid visit IDs or probability class axis")
    p = probabilities.to_numpy(dtype=float)
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-7, rtol=0)):
        raise ValueError("Expected finite normalized visit probabilities")
    visits = roster.groupby("person", sort=True)
    people = visits[["cohort", "group", "y"]].first()
    counts = visits.size()
    received = roster.index.isin(probabilities.index)
    n_pred = roster.loc[received].groupby("person").size().reindex(people.index, fill_value=0)
    coverage = dict(eligible_visits=len(roster), predicted_visits=len(probabilities),
                    eligible_people=len(people), complete_people=int((counts == n_pred).sum()),
                    partial_people=int(((n_pred > 0) & (n_pred < counts)).sum()),
                    missing_people=int((n_pred == 0).sum()), groups=people.group.nunique(),
                    visits_per_person=dict(sorted(Counter(counts).items())),
                    class_people={c: int((people.y == i).sum()) for i, c in enumerate(classes)},
                    class_groups={c: people.loc[people.y == i, "group"].nunique()
                                  for i, c in enumerate(classes)})
    means = (pd.DataFrame(p, index=probabilities.index, columns=classes).reindex(roster.index)
             .groupby(roster.person).mean().reindex(people.index))
    return _Panel(people, means, coverage)


def participant_probabilities(roster, probabilities, classes):
    """Return all participant means indexed by person, in the frozen class order."""
    _check_roster(roster, classes)
    panel = _aggregate(roster, probabilities, classes)
    if not panel.complete:
        raise ValueError("Incomplete eligible visit predictions")
    return panel.probabilities


def select_inner_oof_threshold(training_roster, classes, folds):
    """Each fold supplies (actual complete-pipeline training groups, held-out probabilities).

    training_roster is the current parent training partition only. Complete
    group closures must be supplied from the upstream fit/split receipts.
    """
    _check_roster(training_roster, classes)
    if len(classes) != 2:
        raise ValueError("Multiclass uses fixed argmax")
    all_groups = set(training_roster.group)
    parts = []
    for fit_groups, held_probs in folds:
        held = training_roster.loc[training_roster.index.isin(held_probs.index)]
        _aggregate(held, held_probs, classes)
        held_groups = set(held.group)
        if (not held_groups or set(fit_groups) != all_groups - held_groups
                or len(held) != training_roster.group.isin(held_groups).sum()
                or training_roster.loc[~training_roster.group.isin(held_groups), "y"].nunique() != 2):
            raise ValueError("Invalid inner OOF group partition or training class support")
        parts.append(held_probs)
    if not parts:
        raise ValueError("Missing inner OOF probabilities")
    panel = _aggregate(training_roster, pd.concat(parts), classes)
    y = panel.people.y.to_numpy()
    if not panel.complete or len(np.unique(y)) != 2:
        raise ValueError("Need complete inner OOF coverage of both classes")
    scores = panel.probabilities.iloc[:, 1].to_numpy()
    candidates = np.unique(np.r_[0., scores, .5, 1.])
    pos, neg = int(y.sum()), int((y == 0).sum())

    def rank(t):
        called = scores >= t
        # Integer numerator of balanced accuracy, common denominator 2*pos*neg.
        return (int(y[called].sum()) * neg + int((y[~called] == 0).sum()) * pos,
                -abs(float(t) - .5), t)

    return float(max(candidates, key=rank))


def _decisions(p, threshold):
    if p.shape[1] > 2:
        if threshold is not None:
            raise ValueError("Multiclass decision rule requires threshold=None")
        return np.argmax(p, axis=1)
    if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Provide a frozen binary threshold in [0, 1]")
    return (p[:, 1] >= threshold).astype(int)


def _reliability(score, event):
    bins = np.minimum((score * 10).astype(int), 9)
    rows = []
    for index in range(10):
        mask = bins == index
        n = int(mask.sum())
        rows.append(dict(bin=index, people=n, predicted=float(score[mask].mean()) if n else None,
                         observed=float(event[mask].mean()) if n else None))
    return rows


def _metrics(panel, classes, threshold):
    y = panel.people.y.to_numpy()
    p = panel.probabilities.to_numpy()
    decision = _decisions(p, threshold)
    per_class = {}
    for i, name in enumerate(classes):
        event = y == i
        n = int(event.sum())
        tp = int((decision[event] == i).sum())
        predicted = int((decision == i).sum())
        per_class[name] = dict(
            people=n, called=predicted, correct=tp, recall=tp / n if n else None,
            mean_assigned_probability=float(p[event, i].mean()) if n else None,
            f1=2 * tp / (n + predicted) if n else None,
            roc_auc=float(roc_auc_score(event, p[:, i])) if 0 < n < len(y) else None,
            pr_auc=float(average_precision_score(event, p[:, i])) if 0 < n < len(y) else None,
            calibration=_reliability(p[:, i], event))
    full_support = all(row["people"] for row in per_class.values())
    auc_rows = list(per_class.values()) if len(classes) > 2 else [per_class[classes[1]]]
    brier = ((p[:, 1] - y)**2 if len(classes) == 2
             else np.square(p - np.eye(len(classes))[y]).sum(axis=1))
    return dict(
        roc_auc=float(np.mean([v["roc_auc"] for v in auc_rows])) if full_support else None,
        pr_auc=float(np.mean([v["pr_auc"] for v in auc_rows])) if full_support else None,
        balanced_accuracy=float(np.mean([v["recall"] for v in per_class.values()])) if full_support else None,
        macro_f1=float(np.mean([v["f1"] for v in per_class.values()])) if full_support else None,
        brier=float(brier.mean()), by_class=per_class)


def cohort_report(roster, predictions, classes, thresholds):
    """Report one endpoint/cohort/split. Include every planned model as a key.

    Empty probability frames represent a missing model. Pass an empty eligible
    roster for a registered cohort with zero eligible participants.
    """
    _check_roster(roster, classes)
    if roster.cohort.nunique() > 1:
        raise ValueError("Evaluate one held-out cohort at a time")
    if not predictions or predictions.keys() != thresholds.keys():
        raise ValueError("Supply the decision rule for each model")
    report = {}
    for model, probs in predictions.items():
        panel = _aggregate(roster, probs, classes)
        report[model] = dict(coverage=panel.coverage,
                             status="incomplete" if not panel.complete else
                                    "no eligible visits" if not len(roster) else "complete",
                             metrics=_metrics(panel, classes, thresholds[model])
                             if len(roster) and panel.complete else None)
    return report


def equal_cohort_auc(reports, model, *, expected_cohorts):
    """Pass reports keyed by cohort and the separately frozen cohort roster."""
    expected = tuple(expected_cohorts)
    if len(set(expected)) != len(expected) or set(reports) != set(expected):
        raise ValueError("Reports differ from the registered cohort set")
    values = []
    for cohort in expected:
        entry = reports[cohort][model]
        if all(entry["coverage"]["class_people"].values()):
            if entry["metrics"] is None:
                return None
            values.append(entry["metrics"]["roc_auc"])
    return float(np.mean(values)) if values else None


def _rank_counts(scores):
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    return order, starts


def _weighted_auc(event, weights, ranks):
    order, starts = ranks
    positives = np.add.reduceat((weights * event)[order], starts)
    negatives = np.add.reduceat((weights * ~event)[order], starts)
    # Count all concordant positive/negative pairs with half credit for ties.
    return float(positives @ (np.cumsum(negatives) - .5 * negatives)
                 / (float(positives.sum()) * negatives.sum()))


def primary_contrasts(roster, predictions, classes, thresholds, *, draws=2000, seed=20261124):
    """Seven within-cohort paired differences. Positive means the first model is higher."""
    if not isinstance(draws, int) or draws < 2:
        raise ValueError("Expected at least two bootstrap draws")
    _check_roster(roster, classes)
    if roster.cohort.nunique() > 1:
        raise ValueError("Bootstrap one cohort at a time")
    models = [(arm, head) for arm in ARMS for head in HEADS]
    panels = {m: _aggregate(roster, predictions[m], classes) for m in models}
    people = panels[models[0]].people
    y = people.y.to_numpy()
    group_ids, group = np.unique(people.group, return_inverse=True)
    n_groups = len(group_ids)
    present = np.unique(y)
    full = len(present) == len(classes)
    metrics = ["roc_auc"] if full else [f"recall/{classes[i]}" for i in present]
    result = {}
    for left, right in PRIMARY_PAIRS:
        ready = panels[left].complete and panels[right].complete
        result[left, right] = dict(status="incomplete" if not ready else
                                   "no eligible visits" if not len(y) else "complete",
                                   groups=n_groups, metrics={})
    if not len(y):
        return result
    # Use identical draws for all seven contrasts. Store only scalar paired deltas.
    if full:
        axes = [1] if len(classes) == 2 else range(len(classes))
    prepared = {}
    for model, panel in panels.items():
        if panel.complete:
            p = panel.probabilities.to_numpy()
            calls = _decisions(p, thresholds[model])
            prepared[model] = ([_rank_counts(p[:, i]) for i in axes] if full else calls)

    def estimates(weights):
        values = {}
        support = np.bincount(y, weights=weights, minlength=len(classes))
        for model, table in prepared.items():
            if full:
                if np.all(support):
                    values[model] = [np.mean([_weighted_auc(y == i, weights, rank)
                                              for i, rank in zip(axes, table)])]
            else:
                values[model] = [weights[(y == i) & (table == i)].sum() / support[i]
                                 if support[i] else np.nan for i in present]
        return values

    point = estimates(np.ones(len(y), dtype=int))
    samples = {}
    for pair in PRIMARY_PAIRS:
        a, b = pair
        if a in point and b in point:
            samples[pair] = np.full((draws, len(metrics)), np.nan)
    rng = np.random.default_rng(seed)
    if n_groups >= 2 and samples:
        for r in range(draws):
            counts = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
            values = estimates(counts[group])
            for (a, b), delta in samples.items():
                if a in values and b in values:
                    delta[r] = np.subtract(values[a], values[b])
    for (a, b), delta in samples.items():
        for j, metric in enumerate(metrics):
            valid = delta[np.isfinite(delta[:, j]), j]
            result[a, b]["metrics"][metric] = dict(
                difference=float(point[a][j] - point[b][j]), valid_draws=len(valid),
                attempted_draws=draws if n_groups >= 2 else 0,
                interval=np.quantile(valid, [.025, .975]).tolist()
                if len(valid) >= .9 * draws else None)
    return result
