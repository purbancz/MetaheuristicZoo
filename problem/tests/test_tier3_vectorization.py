"""Certification of the Tier-3 vectorized benchmark problems.

Each reference below is the original pure-Python loop implementation, kept
verbatim. The vectorized evaluate() must agree to tight tolerance across
dimensions (abs floor matters where values approach 0 near optima).

Covered (the remaining loop-based problems of the active evaluation suite):
Michalewicz, RosenbrockModified02, StyblinskiTang, AlpineN1, Salomon,
SchwefelN6, SchwefelN20, SchwefelN36.
"""

import math

import numpy as np
import pytest
from jmetal.core.solution import FloatSolution

from problem.n_variables.alpine import AlpineN1
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.rosenbrock import RosenbrockModified02
from problem.n_variables.salomon import Salomon
from problem.n_variables.schwefel import SchwefelN6, SchwefelN20, SchwefelN36
from problem.n_variables.styblinski import StyblinskiTang


# ---------------------------------------------------------------------------
# Original loop implementations, verbatim.
# ---------------------------------------------------------------------------

def _michalewicz_reference(x, m):
    return -sum(math.sin(xi) * (math.sin(i * xi ** 2 / math.pi)) ** (2 * m)
                for i, xi in enumerate(x, 1))


def _rosenbrock_mod02_reference(x):
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * math.sqrt(abs(x[i + 1] - x[i] ** 2)) + (1.0 - x[i]) ** 2
    return total


def _styblinski_tang_reference(x):
    return 0.5 * sum(xi**4 - 16.0 * xi**2 + 5.0 * xi for xi in x)


def _alpine_n1_reference(x):
    return sum(abs(xi * math.sin(xi) + 0.1 * xi) for xi in x)


def _salomon_reference(x):
    norm = math.sqrt(sum(xi ** 2 for xi in x))
    return 1 - math.cos(2 * math.pi * norm) + 0.1 * norm


def _schwefel_n6_reference(x):
    sum_x = sum(x)
    sum_abs = sum(abs(xi) for xi in x)
    return abs(sum_x) + sum_abs


def _schwefel_n20_reference(x):
    cumulative = 0.0
    max_val = -float("inf")
    for xi in x:
        cumulative += xi
        max_val = max(max_val, abs(cumulative))
    return max_val


def _schwefel_n36_reference(x):
    return sum((418.9829 - xi * math.sin(math.sqrt(abs(xi)))) ** 2 for xi in x)


CASES = [
    (Michalewicz, lambda x: _michalewicz_reference(x, 10)),
    (RosenbrockModified02, _rosenbrock_mod02_reference),
    (StyblinskiTang, _styblinski_tang_reference),
    (AlpineN1, _alpine_n1_reference),
    (Salomon, _salomon_reference),
    (SchwefelN6, _schwefel_n6_reference),
    (SchwefelN20, _schwefel_n20_reference),
    (SchwefelN36, _schwefel_n36_reference),
]


def _evaluate(problem, variables):
    s = FloatSolution(problem.lower_bound, problem.upper_bound, 1, 0)
    s.objectives = [0.0]
    s.variables = list(variables)
    return problem.evaluate(s).objectives[0]


@pytest.mark.parametrize("problem_class,reference",
                         CASES, ids=[c.__name__ for c, _ in CASES])
@pytest.mark.parametrize("dim", [2, 10, 100, 1000])
def test_vectorized_matches_loop_reference(problem_class, reference, dim):
    problem = problem_class(dim)
    lo = np.asarray(problem.lower_bound, dtype=float)
    hi = np.asarray(problem.upper_bound, dtype=float)
    rng = np.random.default_rng(dim)
    for _ in range(3):
        x = list(rng.uniform(lo, hi))
        assert _evaluate(problem, x) == pytest.approx(
            reference(x), rel=1e-12, abs=1e-9)


def test_known_optima_still_hold():
    # Zero-vector optima of the separable/norm-based problems.
    for cls in (AlpineN1, Salomon, SchwefelN6, SchwefelN20):
        assert _evaluate(cls(30), [0.0] * 30) == pytest.approx(0.0, abs=1e-12)
    # Styblinski-Tang: f = -39.16616570377142 * d at x_i = -2.903534018185960.
    assert _evaluate(StyblinskiTang(30), [-2.903534018185960] * 30) == \
        pytest.approx(-39.16616570377142 * 30, rel=1e-12)
    # RosenbrockModified02: f(1, ..., 1) = 0.
    assert _evaluate(RosenbrockModified02(30), [1.0] * 30) == pytest.approx(0.0, abs=1e-12)
