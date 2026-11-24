"""Unequal visit counts, class masses and a final singleton minibatch."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import h0


def panel():
    # 65 visits, eight people, three classes (2,3,3 people).
    counts = np.array([32, 1, 13, 5, 1, 7, 4, 2])
    people = np.repeat(np.arange(len(counts)), counts)
    by_person = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    labels = by_person[people]
    diagrams = [np.array([.04*(i%13) - .32, .12*(p-3)]) for i,p in enumerate(people)]
    return diagrams, labels, people


class ParticipantWeightTests(unittest.TestCase):
    def test_observation_coefficients_and_singleton_batch(self):
        diagrams, labels, people = panel()
        _, _, w, wc = h0._objective_weights(labels, people, torch.float64)
        by_person = [np.flatnonzero(people == p) for p in np.unique(people)]
        np.testing.assert_allclose([float(w[ix].sum()) for ix in by_person], 1/8)
        np.testing.assert_allclose([float(wc[labels == c].sum()) for c in range(3)], 1/3)
        for ix in by_person:
            c = labels[ix[0]]
            participants_in_class = len(set(people[labels == c]))
            self.assertAlmostEqual(float(wc[ix].sum()), 1/(3*participants_in_class))

        # The historical nonlinear batch denominators would give a singleton a
        # unit batch coefficient, irrespective of its class and its visit count.
        tail = torch.tensor([0])
        self.assertLess(float(3*w[tail].sum()), .02)
        self.assertLess(float(3*wc[tail].sum()), .02)

    def test_optimizer_receives_full_epoch_gradient(self):
        diagrams, labels, people = panel()
        by_person = [np.flatnonzero(people == p) for p in np.unique(people)]
        for train, name, dtype, steps in ((h0.learn_image, "full", torch.float64, 1),
                                         (h0.fit_historical, "minibatch", torch.float32, 3)):
            with self.subTest(fit=name):
                self._check_training_path(train, dtype, steps, diagrams, labels, by_person, people)

    def _check_training_path(self, train, dtype, steps, diagrams, labels, by_person, people):
        captures = []
        parameters = []
        class FixedParameters:
            def __init__(self, params, **kwargs):
                parameters.extend(params)
                # Separate centers so all four losses have a nonzero gradient.
                with torch.no_grad():
                    for i, center in enumerate(parameters[1]):
                        center.add_(.12*i*i)
            def zero_grad(self):
                for p in parameters:
                    p.grad = None
            def step(self):
                captures.append([p.grad.detach().double().clone() for p in parameters])
        with patch.object(torch.optim, "Adam", FixedParameters):
            train(diagrams, labels, people, quantile_step=25, epochs=1, seed=23)
        self.assertEqual(len(captures), steps)
        # Independent expression: means of visit-level losses within people,
        # then means across people (and over classes for cross entropy).
        fitted_basis = h0.Image.initial(diagrams, 25)
        bases = torch.tensor(np.stack([fitted_basis.basis(d) for d in diagrams]), dtype=torch.float64)
        raw, centers, weight, bias = [p.detach().double().requires_grad_() for p in parameters]
        z = torch.einsum("k,skj->sj", F.softplus(raw)+1e-8, bases)
        y = torch.tensor(labels)
        cross_entropy = -torch.log_softmax(z@weight.T+bias, dim=1)[torch.arange(len(y)), y]
        own = torch.sum((z-centers[y])**2, dim=1)
        pairs = [(i,j) for i in range(3) for j in range(i+1,3)]
        distances = {(i,j):torch.sqrt(((centers[i]-centers[j])**2).sum()+1e-12) for i,j in pairs}
        nearest = torch.stack([torch.stack([distances[key] for key in pairs if c in key]).min()
                               for c in range(3)])
        radius = torch.clamp(torch.sqrt(own+1e-12)-nearest[y]/2, min=0)**2
        classes = [labels[ids[0]] for ids in by_person]
        ce = torch.stack([torch.stack([cross_entropy[ids].mean() for ids,c in zip(by_person,classes)
                                       if c == label]).mean() for label in range(3)]).mean()
        geometric = torch.stack([(own[ids]+radius[ids]).mean() for ids in by_person]).mean()
        between = torch.stack([torch.clamp(10-d,min=0)**4 for d in distances.values()]).mean()
        independent = torch.autograd.grad(1000*ce + geometric + 200*between, (raw, centers, weight, bias))
        for j, expect in enumerate(independent):
            actual = torch.stack([x[j] for x in captures]).mean(0)
            torch.testing.assert_close(actual, expect, rtol=3e-6 if dtype==torch.float32 else 2e-12,
                                       atol=.06 if dtype==torch.float32 and j==1 else 2e-4 if dtype==torch.float32 else 2e-9)

    def test_epoch_exact_while_batch_composition_varies(self):
        _, labels, people = panel()
        _, _, w, wc = h0._objective_weights(labels, people, torch.float64)
        per_visit = torch.tensor(np.sin(np.arange(len(labels))*.15), dtype=torch.float64)
        realized = []
        for seed in (23, 47, 71, 97):
            batches = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed)).split(32)
            # Fixed parameters; the two types of contributions have different weights.
            pieces = torch.stack([len(batches)*((wc[idx]+w[idx])*per_visit[idx]).sum() for idx in batches])
            torch.testing.assert_close(pieces.mean(), ((wc+w)*per_visit).sum(), atol=1e-14, rtol=0)
            realized.append(float(pieces[0]))
        self.assertGreater(np.std(realized), .01)

    def test_bound_estimator_matches_pooled_training_death_percentiles(self):
        # Historical percentiles pool deaths. The training-only cut can shift when
        # a visit with many deaths is replicated, independently of the loss weights.
        raw = [[.15, .22, .35, .4], [.72], [.95]]
        diagrams = [np.log(-np.log(np.array(xs))) for xs in raw]
        pooled = np.percentile(np.concatenate(raw), [25,50,75])
        fitted = h0.Image.initial(diagrams, 25)
        np.testing.assert_allclose(fitted.bounds, [0,*pooled,1], rtol=0, atol=4e-15)
        duplicated = h0.Image.initial(diagrams + [diagrams[0]], 25)
        self.assertLess(duplicated.bounds[2], fitted.bounds[2])
        # Held-out application consumes the frozen bounds.
        fitted.transform([np.array([-10., 15.])])
        np.testing.assert_allclose(fitted.bounds, [0,*pooled,1], rtol=0, atol=4e-15)


if __name__ == "__main__":
    unittest.main()
