"""A-contrario significance of a matched region (Number of False Alarms).

Per-patch tests can't tell a copied region from luck: a 3x3 patch matches some
unrelated patch somewhere surprisingly often. What luck can't easily do is make
many neighbouring patches agree on the *same* offset and the *same* color
transform. The a-contrario framework (Desolneux, Moisan & Morel) turns that
into a number:

    NFA = N_tests * P[Binomial(n, p) >= k]

``n`` source patches were tested inside the region, ``k`` of them agree with
the region's offset and color transform, and ``p`` is the chance that a single
unrelated patch agrees by accident. ``N_tests`` counts how many region
hypotheses we implicitly tried. NFA is the expected number of regions at least
this good in data with no copy at all; NFA < 1 is the usual detection rule and
the value itself is something you can defend in a report.

Two approximations, both on the conservative side:

* Overlapping patches aren't independent. We divide counts by the overlap
  factor (patch_size / stride)^2, so only roughly disjoint patches are counted.
* ``p`` is estimated from the observed rate of verified matches, which includes
  the true ones, so it overestimates the chance rate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def overlap_factor(patch_size: int, stride: int) -> float:
    """How many stride-spaced patches share pixels with one patch, per axis squared."""
    return max(1.0, (patch_size / max(stride, 1)) ** 2)


def log10_nfa(k: float, n: float, p: float, n_tests: float) -> float:
    """log10 of the number of false alarms for k successes out of n at rate p.

    k and n may be fractional (after overlap correction); they're rounded
    conservatively (k down, n up).
    """
    k_int = int(math.floor(k))
    n_int = max(int(math.ceil(n)), k_int)
    log_tests = math.log10(max(n_tests, 1.0))
    if k_int <= 0:
        return log_tests
    p = float(np.clip(p, 1e-300, 1.0))
    log_tail = stats.binom.logsf(k_int - 1, n_int, p)
    if not np.isfinite(log_tail):
        # The tail underflowed. P[X = k] is the dominant term that far out, so
        # it's off by a small factor at most - irrelevant at these magnitudes.
        log_tail = stats.binom.logpmf(k_int, n_int, p)
    return log_tests + float(log_tail) / math.log(10.0)
