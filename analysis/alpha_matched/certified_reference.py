"""Small directed-graph Ricci validation oracle. Input weights are positive arm support.

Exact fractions validate first-hit residuals, full transport marginals and primal/dual
objectives. MP interval elementary functions enclose non-rational weights and costs.
The reference mode deliberately spends memory and arithmetic for auditability.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json
import math
import platform
import mpmath

from mpmath.ctx_iv import MPIntervalContext
from mpmath.libmp import to_rational
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, vstack, kron
from scipy.sparse.linalg import splu

IV = MPIntervalContext()
IV.dps = 75
ZERO = F(0)
ONE = F(1)


def value(x):
    """Decimal strings denote exact decimals. Floats denote their exact binary values."""
    if isinstance(x, bool):
        raise ValueError("boolean is not a numerical input")
    return F(x)


def exact_interval(x):
    return IV.mpf(x.numerator) / x.denominator


@dataclass(frozen=True)
class Band:
    lo: F
    hi: F

    def __post_init__(self):
        if self.lo > self.hi:
            raise ArithmeticError("reversed interval")

    def mid(self):
        return (self.lo + self.hi) / 2

    def radius(self):
        return (self.hi - self.lo) / 2


def enclosed(x):
    return Band(*(F(*to_rational(endpoint)) for endpoint in x._mpi_))


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def directed_support(vertices, edges):
    nodes = sorted(vertices)
    if not all(isinstance(node, str) for node in nodes) or len(nodes) != len(set(nodes)):
        raise ValueError("vertices must have distinct string identifiers")
    index = {node: i for i, node in enumerate(nodes)}
    active = {}
    for (source, target), r in edges.items():
        if source not in index or target not in index or source == target:
            raise ValueError("expected endpoints in vertex set and nonself projected pairs")
        r = value(r)
        if r < 0:
            raise ValueError("negative support")
        if r:
            active[index[source], index[target]] = r
    return nodes, dict(sorted(active.items()))


def response(r, family, parameter):
    if family in ("linear", "uniform"):
        if parameter is not None:
            raise ValueError("unexpected transition parameter")
        weight = r if family == "linear" else ONE
    elif family == "saturation":
        k = value(parameter)
        if k <= 0:
            raise ValueError("K must be positive")
        weight = r / (k + r)
    elif family == "power":
        beta = value(parameter)
        if beta <= 0:
            raise ValueError("beta must be positive")
        if beta.denominator != 1:
            return enclosed(exact_interval(r) ** exact_interval(beta))
        weight = r ** beta.numerator
    else:
        raise ValueError("unknown transition family")
    return Band(weight, weight)


def transitions(n, edges, family, parameter):
    # Validate configuration even on empty graphs.
    response(ONE, family, parameter)
    weights = [{} for _ in range(n)]
    for (i, j), r in edges.items():
        weights[i][j] = response(r, family, parameter)
    rows, measures, tv = [], [], []
    for i, row in enumerate(weights):
        k = len(row)
        if not k:
            rows.append({i: ONE})
            measures.append({i: ONE})
            tv.append(ZERO)
            continue
        lo, hi = sum(w.lo for w in row.values()), sum(w.hi for w in row.values())
        total = (lo + hi) / 2
        center = {j: w.mid() / total for j, w in row.items()}
        rows.append(center)
        measures.append({i: F(1, k + 1), **{j: q * F(k, k + 1) for j, q in center.items()}})
        error = sum(max(p - row[j].lo / hi, row[j].hi / lo - p) for j, p in center.items())
        tv.append(min(ONE, error / 2))
    return rows, measures, tv, [F(len(w), len(w) + 1) * e for w, e in zip(weights, tv)]


def gauss_solve(a, b):
    """Elimination over exact rationals for small first-hit Dirichlet systems."""
    a = [r.copy() + [rhs] for r, rhs in zip(a, b)]
    n = len(a)
    for j in range(n):
        pivot = next(i for i in range(j, n) if a[i][j])
        a[j], a[pivot] = a[pivot], a[j]
        for i in range(j + 1, n):
            if a[i][j]:
                ratio = a[i][j] / a[j][j]
                for col in range(j, n + 1):
                    a[i][col] -= ratio * a[j][col]
    sol = [ZERO] * n
    for i in reversed(range(n)):
        sol[i] = (a[i][n] - sum(a[i][j] * sol[j] for j in range(i + 1, n))) / a[i][i]
    return sol


def exact_column(rows, target, rho):
    other = [i for i in range(len(rows)) if i != target]
    a = [[F(i == j) - rho * rows[i].get(j, ZERO) for j in other] for i in other]
    b = [rho * rows[i].get(target, ZERO) for i in other]
    result = gauss_solve(a, b)
    result.insert(target, ONE)
    return result


def sparse_factor(rows, rho):
    n = len(rows)
    entries = [(i, j, float(p)) for i, row in enumerate(rows) for j, p in row.items()]
    row, col, data = zip(*entries)
    p = csc_matrix((data, (row, col)), shape=(n, n))
    return splu(eye(n, format="csc") - float(rho) * p)


def distance(q, epsilon):
    return enclosed(-IV.ln(exact_interval(epsilon + (ONE - epsilon) * q)))


def first_hits(rows, row_tv, rho, epsilon, method):
    n = len(rows)
    if method not in ("exact", "sparse"):
        raise ValueError("unknown first-hit solver")
    factor = sparse_factor(rows, rho) if n and method == "sparse" else None
    cost, hits = {}, {}
    max_residual = ZERO
    for target in range(n):
        if method == "exact":
            h = exact_column(rows, target, rho)
        else:
            unit = np.zeros(n)
            unit[target] = 1
            z = factor.solve(unit)
            if not np.all(np.isfinite(z)) or z[target] <= 0:
                raise ArithmeticError("invalid resolvent")
            h = [F(float(x)) for x in z / z[target]]
            for u in range(n):
                if u != target and (h[u] < F("-1e-12") or h[u] > rho + F("1e-12")):
                    raise ArithmeticError("substantive first-hit range violation")
                h[u] = ONE if u == target else max(ZERO, min(rho, h[u]))
        residual = max((abs(h[i] - rho * sum(p * h[j] for j, p in row.items())) + rho * row_tv[i]
                        for i, row in enumerate(rows) if i != target), default=ZERO)
        max_residual = max(max_residual, residual)
        delta = residual / (1 - rho)
        for u in range(n):
            if u == target:
                hits[u, target], cost[u, target] = Band(ONE, ONE), Band(ZERO, ZERO)
                continue
            hits[u, target] = Band(max(ZERO, h[u] - delta), min(rho, h[u] + delta))
            bound = hits[u, target]
            cost[u, target] = Band(distance(bound.hi, epsilon).lo, distance(bound.lo, epsilon).hi)
    return hits, cost, max_residual


def transport(a, b, c):
    """Enclose optimum for rational marginals and interval costs on all cells."""
    nr, nc = len(a), len(b)
    matrix = vstack([kron(eye(nr), np.ones((1, nc))), kron(np.ones((1, nr)), eye(nc))], format="csc")
    result = linprog([float(cell.mid()) for row in c for cell in row], A_eq=matrix,
                     b_eq=[float(x) for x in a+b], method="highs-ds",
                     options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.eqlin.marginals).all():
        raise ArithmeticError("transport LP has no finite proposal")
    plan = [[ZERO] * nc for _ in range(nr)]
    left, right = a.copy(), b.copy()
    proposal = result.x.reshape(nr, nc)
    margin_error = max([abs(sum(F(float(x)) for x in proposal[i]) - a[i]) for i in range(nr)]
                       + [abs(sum(F(float(x)) for x in proposal[:,j]) - b[j]) for j in range(nc)])
    if margin_error > F("1e-8") or np.min(proposal) < -1e-10:
        raise ArithmeticError("transport proposal violates marginal/nonnegativity limits")
    for i in range(nr):
        for j in range(nc):
            flow = min(left[i], right[j], max(ZERO, F(float(proposal[i, j]))))
            plan[i][j] = flow
            left[i] -= flow
            right[j] -= flow
    # Complete all residual mass exactly, including masses smaller than binary64.
    for i in range(nr):
        for j in range(nc):
            flow = min(left[i], right[j])
            plan[i][j] += flow
            left[i] -= flow
            right[j] -= flow
    if any(left + right) or any(x < 0 for row in plan for x in row):
        raise ArithmeticError("could not construct an exactly feasible transport witness")
    upper = sum(plan[i][j] * c[i][j].hi for i in range(nr) for j in range(nc))
    f = [F(float(x)) for x in result.eqlin.marginals[:nr]]
    g = [min(c[i][j].lo - f[i] for i in range(nr)) for j in range(nc)]
    # Tighten every row after making columns feasible. Tiny rows can be absent
    # from the floating basis, but their dual inequalities still constrain all columns.
    f = [min(c[i][j].lo - g[j] for j in range(nc)) for i in range(nr)]
    lower = sum(v * w for v, w in zip(a, f)) + sum(v * w for v, w in zip(b, g))
    return Band(max(ZERO, lower), upper), margin_error


def upper_float(x):
    y = float(x)
    return math.nextafter(y, math.inf) if F(y) < x else y


def lower_float(x):
    return -upper_float(-x)


def geometry(vertices, support, family="linear", parameter=None, rho="0.9", epsilon="0.0001",
             method="exact", tolerance="1e-7"):
    """Return certified all-edge records or raise. No partial feature vector is returned."""
    rho, epsilon, tolerance = map(value, (rho, epsilon, tolerance))
    if not (0 < rho < 1 and 0 < epsilon < 1 and tolerance > 0):
        raise ValueError("invalid horizon, cost regularization or numerical tolerance")
    nodes, edges = directed_support(vertices, support)
    rows, mus, tv, mu_tv = transitions(len(nodes), edges, family, parameter)
    _, costs, residual = first_hits(rows, tv, rho, epsilon, method)
    M = distance(ZERO, epsilon).hi
    records, max_error = [], ZERO
    for i, j in edges:
        x, y = list(mus[i]), list(mus[j])
        c = [[costs[u,v] for v in y] for u in x]
        w, margin_error = transport(list(mus[i].values()), list(mus[j].values()), c)
        perturbation = M * (mu_tv[i] + mu_tv[j])
        lower, upper = max(ZERO, w.lo - perturbation), min(M, w.hi + perturbation)
        d = costs[i,j]
        if d.lo <= 0:
            raise ArithmeticError("nonpositive intervertex distance")
        k = Band(1 - upper / d.lo, 1 - lower / d.hi)
        center = 0.0 if k.lo <= 0 <= k.hi else float(k.mid())
        error = max(k.hi - F(center), F(center) - k.lo)
        if error > tolerance or not math.isfinite(center):
            raise ArithmeticError(f"curvature tolerance not met for {(nodes[i], nodes[j])}")
        max_error = max(max_error, error)
        records.append(dict(source=nodes[i], target=nodes[j], B=1, K0=center,
                            kappa_bounds=[lower_float(k.lo), upper_float(k.hi)],
                            d_bounds=[lower_float(d.lo), upper_float(d.hi)],
                            W_bounds=[lower_float(lower), upper_float(upper)],
                            proposal_max_marginal_error=upper_float(margin_error), certified_marginal_error=0))
    active_pairs = [[nodes[i], nodes[j]] for i, j in edges]
    cfg = {"family":family, "parameter":None if parameter is None else str(value(parameter)),
           "rho":str(rho), "epsilon":str(epsilon), "method":method, "tolerance":str(tolerance),
           "precision_decimal_digits":IV.dps, "scipy":scipy.__version__, "numpy":np.__version__,
           "mpmath":mpmath.__version__, "python":platform.python_version(),
           "implementation_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    return dict(active_edges=len(edges), finite_edges=len(records), failed_nonfinite=0,
                activity_sha256=digest([nodes,active_pairs]),
                support_sha256=digest([nodes, [[nodes[i],nodes[j],str(r)] for (i,j),r in edges.items()]]),
                solver_configuration=cfg, solver_sha256=digest(cfg), records=records,
                max_first_hit_residual_bound=upper_float(residual),
                max_curvature_absolute_error_bound=upper_float(max_error))


def encode(axis, geometry_record):
    """B is carried independently of the value of K0, including K0=0."""
    if len(axis) != len(set(axis)):
        raise ValueError("duplicate classifier coordinate")
    records = geometry_record["records"]
    values = {(r["source"], r["target"]): r["K0"] for r in records}
    n = len(values)
    if (geometry_record["active_edges"] != n or geometry_record["finite_edges"] != n
            or len(records) != n or geometry_record["failed_nonfinite"] != 0
            or any(r["B"] != 1 or not math.isfinite(r["K0"]) for r in records)):
        raise ValueError("incomplete geometry")
    return [int(e in values) for e in axis], [values.get(e, 0.0) for e in axis]
