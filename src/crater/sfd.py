"""Crater size-frequency distribution (SFD) and the relative (R) plot.

Definitions and units (fixed here, exported in the CSV header)
-------------------------------------------------------------
Diameters enter in **metres** (the project's working unit, DECISIONS.md D-003
approves 20 m to 1000 m).  The table is reported in **kilometres**, which is the
convention of the crater-counting literature and the unit in which ``R`` is
usually quoted::

    D_low, D_high   bin edges                                     [km]
    D_g             geometric mean diameter, sqrt(D_low * D_high)  [km]
    Delta_D         D_high - D_low                                 [km]
    N               unique included crater count in the bin        [-]
    A               usable counting area for the bin               [km^2]
    R               D_g**3 * N / (A * Delta_D)                     [-]

``R`` is dimensionless: ``km**3 / (km**2 * km)``.  Mixing metres into one factor
and kilometres into another changes ``R`` by 10**3 or more, so the units are
asserted by ``tests/test_sfd.py`` rather than trusted.

Why R, and why it should be flat
--------------------------------
``R`` is the crater density relative to a differential power law of slope -3.  If
``dN/dD = C * D**-3`` then over a bin ``[lo, hi)`` the expected count is
``N = C/2 * (lo**-2 - hi**-2)``, and substituting into the definition gives::

    R = C / (2 * A) * (1 + r) / sqrt(r)        with   r = hi / lo

which contains no reference to the bin's absolute diameter.  For log bins of
**constant ratio** ``r``, ``R`` is therefore exactly constant: a flat R-plot is
the signature of a -3 differential slope, and departures from flat are the
signal.  ``tests/test_sfd.py`` validates the whole pipeline against that
identity using a synthetic ``D**-3`` population.

Binning convention
------------------
Bins are a geometric progression ``anchor * base**(j / bins_per_octave)`` on a
grid that depends only on ``(anchor, base, bins_per_octave)`` and not on the
data -- so two series (catalogue vs detector) binned with the same parameters get
**identical** edges and can be compared bin by bin.  Intervals are half-open,
``D_low <= D < D_high`` (:data:`BIN_INTERVAL_CONVENTION`), so a crater measured
exactly on an edge falls in the bin that edge opens.  ``2**(1/4)`` per bin (four
bins per factor of two, i.e. 18 bins per decade) is the common choice; it is a
parameter, not a constant.

Two uncertainties, never one
----------------------------
Two independent error sources are reported and are deliberately **never
combined**:

* **Poisson counting uncertainty** -- the finite number of craters.  Computed as
  an exact Garwood interval from chi-square quantiles
  (:func:`poisson_interval`), which stays correct at ``N = 1, 2, 3``, where a
  ``sqrt(N)`` normal approximation gives a lower bound that is negative or
  absurdly tight.  Columns ``R_poisson_low`` / ``R_poisson_high``.
* **Diameter measurement uncertainty** -- craters migrating between bins.
  Propagated by Monte Carlo over per-crater diameter errors
  (:class:`DiameterUncertainty`), giving a spread in ``N`` per bin.  Columns
  ``N_mc_*`` / ``R_mc_*``.

They answer different questions ("would another sample of craters look like
this?" versus "would another measurement of these craters look like this?"), they
are correlated across bins in completely different ways (the Monte Carlo moves a
crater *out* of one bin and *into* the next; Poisson does not), and adding them
in quadrature into one error bar would make both unrecoverable.  No function in
this module sums them.

Empty bins
----------
A bin with ``N = 0`` has ``R = 0`` exactly, which has no position on a
logarithmic axis.  Nothing here adds an epsilon to make ``log`` work.  Such a
bin gets ``R = NaN``, the flag :data:`FLAG_ZERO_COUNT`, and a finite
``R_poisson_high`` that *is* the upper limit; the plotting code draws it as a
downward caret (INTERFACES.md rule 6).  A bin whose counting area has been
eroded to zero by the edge rule gets ``R = NaN`` and :data:`FLAG_NO_AREA`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: Half-open bins: ``D_low <= D < D_high``.  A crater exactly on an edge belongs
#: to the bin that edge *opens*, never to both and never to neither.
BIN_INTERVAL_CONVENTION = "half_open_low_inclusive"

#: Bins are a geometric progression; this names the convention in the CSV.
BIN_PROGRESSION = "geometric_anchor_base_per_octave"

#: Flags attached to each bin.
FLAG_OK = "ok"
FLAG_ZERO_COUNT = "zero_count_upper_limit"
FLAG_NO_AREA = "no_counting_area"

#: Default RNG seed (``config/project.yaml`` ``seeds.global``).
DEFAULT_SEED = 20261001

#: Poisson and Monte Carlo uncertainties are reported separately, never summed.
UNCERTAINTY_POLICY = "poisson_and_measurement_reported_separately"

_EDGE_SNAP_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
def log_bin_edges_m(
    d_min_m: float,
    d_max_m: float,
    *,
    bins_per_octave: int = 4,
    base: float = 2.0,
    anchor_m: float = 1.0,
) -> np.ndarray:
    """Geometric bin edges in metres, on a data-independent grid.

    Edges are ``anchor_m * base**(j / bins_per_octave)`` for consecutive integer
    ``j``, chosen so that ``edges[0] <= d_min_m`` and ``d_max_m < edges[-1]``.
    Because the grid depends only on ``(anchor_m, base, bins_per_octave)``, two
    catalogues binned with the same parameters -- even over different diameter
    ranges -- share edges wherever they overlap, which is what comparing two
    series on one R-plot requires.

    The upper edge is strictly above ``d_max_m`` so that the half-open
    convention (:data:`BIN_INTERVAL_CONVENTION`) never silently drops the largest
    crater; the price is that the topmost bin may legitimately be empty.

    Parameters
    ----------
    d_min_m, d_max_m
        Diameter range to cover, metres.  Must satisfy ``0 < d_min <= d_max``.
    bins_per_octave
        Number of bins per factor of ``base``.  4 gives the familiar
        ``2**(1/4)`` progression (18 bins per decade); 1 gives ``sqrt(2)``-wide
        octave bins when ``base=2``.
    base
        Factor spanned by ``bins_per_octave`` bins.  2.0 by convention.
    anchor_m
        The grid passes exactly through this diameter.  1 m by default, which
        makes the grid reproducible from the parameters alone.

    Returns
    -------
    ndarray
        ``n_bins + 1`` strictly increasing edges in metres.
    """
    if not (d_min_m > 0.0 and d_max_m > 0.0):
        raise ValueError("diameters must be positive")
    if d_max_m < d_min_m:
        raise ValueError("d_max_m must be >= d_min_m")
    if bins_per_octave < 1:
        raise ValueError("bins_per_octave must be a positive integer")
    if base <= 1.0:
        raise ValueError("base must exceed 1")
    if anchor_m <= 0.0:
        raise ValueError("anchor_m must be positive")

    ratio = base ** (1.0 / bins_per_octave)
    log_ratio = np.log(ratio)
    # +/- tolerance so a d_min that sits exactly on a grid point is not pushed
    # one bin outwards by float round-off.
    j_lo = int(np.floor(np.log(d_min_m / anchor_m) / log_ratio + _EDGE_SNAP_TOL))
    j_hi = int(np.floor(np.log(d_max_m / anchor_m) / log_ratio + _EDGE_SNAP_TOL)) + 1
    return anchor_m * ratio ** np.arange(j_lo, j_hi + 1, dtype=float)


def bin_index(diameters_m, edges_m) -> np.ndarray:
    """Index of the bin each diameter falls in, under the half-open convention.

    Returns ``-1`` for diameters below ``edges[0]`` and ``n_bins`` for diameters
    at or above ``edges[-1]``; those are out of range, not silently clamped.
    """
    d = np.asarray(diameters_m, dtype=float)
    e = np.asarray(edges_m, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges_m must be a 1-D array of at least two edges")
    if np.any(np.diff(e) <= 0):
        raise ValueError("edges_m must be strictly increasing")
    # side="right" puts a value exactly on an edge into the bin that edge opens.
    return np.searchsorted(e, d, side="right") - 1


def bin_counts(diameters_m, edges_m) -> tuple[np.ndarray, int, int]:
    """Counts per bin, plus the number of diameters below and above the range.

    Returns
    -------
    (counts, n_below, n_above)
        ``counts`` has length ``len(edges) - 1``.  Out-of-range craters are
        reported rather than discarded quietly.
    """
    idx = bin_index(diameters_m, edges_m)
    n_bins = len(np.asarray(edges_m)) - 1
    n_below = int(np.count_nonzero(idx < 0))
    n_above = int(np.count_nonzero(idx >= n_bins))
    inside = idx[(idx >= 0) & (idx < n_bins)]
    counts = np.bincount(inside, minlength=n_bins).astype(np.int64)
    return counts, n_below, n_above


# --------------------------------------------------------------------------- #
# Poisson counting uncertainty
# --------------------------------------------------------------------------- #
def poisson_interval(n, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact (Garwood) two-sided Poisson confidence interval on the mean.

    For an observed count ``n`` and confidence ``level = 1 - alpha``::

        lower = chi2.ppf(alpha/2,   2*n)   / 2      (exactly 0 for n = 0)
        upper = chi2.ppf(1-alpha/2, 2*n+2) / 2

    This is the Clopper-Pearson analogue for the Poisson distribution and is the
    interval obtained by inverting the exact test.  It is correct at the low
    counts that dominate the large-diameter end of a crater count, where the
    ``n +/- sqrt(n)`` normal approximation fails badly: at ``n = 1`` it gives
    ``[0, 2]`` (a lower bound of 0 that is really 0.025 and an upper bound of 2
    that is really 5.57), and at ``n = 0`` it gives the meaningless ``[0, 0]``.
    Garwood is conservative by construction -- its coverage is at least the
    nominal level, and because the Poisson distribution is discrete it is
    typically a little above.

    For ``n = 0`` the interval degenerates to ``[0, -ln(alpha/2)]``
    (3.689 at 95 %), i.e. a pure **upper limit**, which the third return value
    flags.  The lower bound 0.0 is the exact bound, not a fabricated value, but
    it has no place on a logarithmic axis: callers must use the flag, not the
    number.

    Parameters
    ----------
    n
        Non-negative integer count(s).
    level
        Two-sided confidence level in (0, 1).

    Returns
    -------
    (low, high, is_upper_limit)
        Arrays shaped like ``n`` (0-d for a scalar input).
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    counts = np.asarray(n)
    if not np.issubdtype(counts.dtype, np.integer):
        if np.any(counts != np.floor(counts)):
            raise ValueError("Poisson counts must be integers")
        counts = counts.astype(np.int64)
    if np.any(counts < 0):
        raise ValueError("Poisson counts must be non-negative")

    alpha = 1.0 - level
    low = stats.chi2.ppf(alpha / 2.0, 2 * counts) / 2.0
    low = np.where(counts == 0, 0.0, low)
    high = stats.chi2.ppf(1.0 - alpha / 2.0, 2 * counts + 2) / 2.0
    is_upper_limit = counts == 0
    if np.ndim(n) == 0:
        return float(low), float(high), bool(is_upper_limit)
    return low, high, is_upper_limit


# --------------------------------------------------------------------------- #
# Diameter measurement uncertainty (kept separate from Poisson)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiameterUncertainty:
    """Per-crater diameter measurement uncertainty, for Monte Carlo propagation.

    The perturbation is **multiplicative and log-normal**::

        D' = D * exp(s * z),      z ~ N(0, 1)
        s  = sqrt(relative_sigma**2 + (absolute_sigma_m / D)**2)

    so a perturbed diameter is always positive and the median diameter is
    preserved.  A Gaussian directly in ``D`` would put a 20 m crater measured to
    +/-20 % at a non-positive diameter in about 0.3 % of draws, which then cannot
    be binned on a log grid at all.

    Attributes
    ----------
    relative_sigma
        Fractional 1-sigma, scalar or per crater (0.1 = 10 %).
    absolute_sigma_m
        Additive 1-sigma in metres, scalar or per crater; combined with
        ``relative_sigma`` in quadrature in log space.  Use it for a
        rim-fitting floor of a couple of pixels.
    n_draws
        Number of Monte Carlo realisations.
    seed
        RNG seed; fixed by default so results are reproducible
        (INTERFACES.md rule 7).
    percentiles
        Percentiles of the per-bin count distribution to report.
    """

    relative_sigma: float | np.ndarray = 0.0
    absolute_sigma_m: float | np.ndarray = 0.0
    n_draws: int = 1000
    seed: int = DEFAULT_SEED
    percentiles: tuple[float, float] = (16.0, 84.0)

    def __post_init__(self) -> None:
        if self.n_draws < 1:
            raise ValueError("n_draws must be at least 1")
        if np.any(np.asarray(self.relative_sigma, float) < 0) or np.any(
            np.asarray(self.absolute_sigma_m, float) < 0
        ):
            raise ValueError("sigmas must be non-negative")
        lo, hi = self.percentiles
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    def log_sigma(self, diameters_m) -> np.ndarray:
        """Combined fractional 1-sigma in log space, per crater."""
        d = np.asarray(diameters_m, dtype=float)
        if np.any(d <= 0):
            raise ValueError("diameters must be positive")
        rel = np.asarray(self.relative_sigma, dtype=float)
        absol = np.asarray(self.absolute_sigma_m, dtype=float)
        return np.sqrt(rel**2 + (absol / d) ** 2) * np.ones_like(d)

    @property
    def is_zero(self) -> bool:
        """True if no uncertainty was specified, so Monte Carlo is a no-op."""
        return bool(
            np.all(np.asarray(self.relative_sigma, float) == 0.0)
            and np.all(np.asarray(self.absolute_sigma_m, float) == 0.0)
        )

    def draw(self, diameters_m, rng: np.random.Generator) -> np.ndarray:
        """One realisation of perturbed diameters, metres."""
        d = np.asarray(diameters_m, dtype=float)
        s = self.log_sigma(d)
        return d * np.exp(s * rng.standard_normal(d.shape))


def monte_carlo_bin_counts(
    diameters_m, edges_m, uncertainty: DiameterUncertainty
) -> np.ndarray:
    """Per-bin counts for every Monte Carlo realisation of the diameters.

    Returns
    -------
    ndarray of shape ``(n_draws, n_bins)``
        Integer counts.  Craters that leave the binned range in a realisation
        are dropped from that realisation, exactly as they would be from the real
        count -- so the row sums need not be constant, and that is physical, not
        a bug.
    """
    d = np.asarray(diameters_m, dtype=float)
    n_bins = len(np.asarray(edges_m)) - 1
    rng = np.random.default_rng(uncertainty.seed)
    out = np.empty((uncertainty.n_draws, n_bins), dtype=np.int64)
    for i in range(uncertainty.n_draws):
        perturbed = uncertainty.draw(d, rng)
        out[i], _, _ = bin_counts(perturbed, edges_m)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SFDTable:
    """One series of a size-frequency distribution, bin by bin.

    Every array has one entry per bin.  Diameters are kilometres, areas square
    kilometres, ``R`` dimensionless (see the module docstring).
    """

    label: str
    d_low_km: np.ndarray
    d_high_km: np.ndarray
    d_g_km: np.ndarray
    delta_d_km: np.ndarray
    n: np.ndarray
    area_km2: np.ndarray
    r: np.ndarray
    r_poisson_low: np.ndarray
    r_poisson_high: np.ndarray
    n_poisson_low: np.ndarray
    n_poisson_high: np.ndarray
    is_upper_limit: np.ndarray
    flag: np.ndarray
    confidence_level: float
    n_below_range: int = 0
    n_above_range: int = 0
    n_mc_mean: np.ndarray | None = None
    n_mc_std: np.ndarray | None = None
    n_mc_low: np.ndarray | None = None
    n_mc_high: np.ndarray | None = None
    r_mc_low: np.ndarray | None = None
    r_mc_high: np.ndarray | None = None
    mc_percentiles: tuple[float, float] | None = None
    mc_n_draws: int | None = None
    edges_m: np.ndarray | None = None

    @property
    def n_bins(self) -> int:
        return len(self.d_low_km)

    @property
    def has_monte_carlo(self) -> bool:
        return self.n_mc_mean is not None

    def to_dataframe(self) -> pd.DataFrame:
        """All columns as a :class:`pandas.DataFrame`, one row per bin."""
        cols = {
            "series": self.label,
            "D_low_km": self.d_low_km,
            "D_high_km": self.d_high_km,
            "D_g_km": self.d_g_km,
            "Delta_D_km": self.delta_d_km,
            "N": self.n,
            "A_km2": self.area_km2,
            "R": self.r,
            "R_poisson_low": self.r_poisson_low,
            "R_poisson_high": self.r_poisson_high,
            "N_poisson_low": self.n_poisson_low,
            "N_poisson_high": self.n_poisson_high,
            "confidence_level": self.confidence_level,
            "is_upper_limit": self.is_upper_limit,
            "flag": self.flag,
        }
        if self.has_monte_carlo:
            lo, hi = self.mc_percentiles
            cols.update(
                {
                    "N_mc_mean": self.n_mc_mean,
                    "N_mc_std": self.n_mc_std,
                    f"N_mc_p{lo:g}": self.n_mc_low,
                    f"N_mc_p{hi:g}": self.n_mc_high,
                    f"R_mc_p{lo:g}": self.r_mc_low,
                    f"R_mc_p{hi:g}": self.r_mc_high,
                    "mc_n_draws": self.mc_n_draws,
                }
            )
        return pd.DataFrame(cols)


def size_frequency_table(
    diameters_m,
    edges_m,
    areas_km2,
    *,
    label: str = "series",
    confidence_level: float = 0.95,
    uncertainty: DiameterUncertainty | None = None,
    crater_ids: Sequence | None = None,
) -> SFDTable:
    """Build the per-bin SFD / R table.

    Parameters
    ----------
    diameters_m
        Diameters of the **included, deduplicated** craters, in metres.  This
        module does not deduplicate (that is ``crater.dedup``'s job); pass
        ``crater_ids`` to have uniqueness checked rather than assumed.
    edges_m
        Bin edges in metres from :func:`log_bin_edges_m`.  Pass the *same* array
        to every series that will be compared.
    areas_km2
        Usable counting area per bin, square kilometres.  Scalar for a
        diameter-independent area, or one value per bin -- which is the
        diameter-dependent ``A_i`` of
        :func:`crater.area.usable_area_km2_by_diameter` under
        :data:`crater.area.EDGE_RULE`.  Must be non-negative; a bin with zero
        area yields ``R = NaN`` and :data:`FLAG_NO_AREA`.
    label
        Series name, used in the plot legend and the CSV ``series`` column.
    confidence_level
        Two-sided level for the Garwood Poisson interval.
    uncertainty
        Optional diameter measurement uncertainty.  Its Monte Carlo spread is
        stored in separate ``*_mc_*`` columns and is **never** combined with the
        Poisson interval.
    crater_ids
        Optional identifiers, one per crater.  Duplicates raise, which is the
        check behind "N = unique included crater count".

    Returns
    -------
    SFDTable
    """
    d = np.asarray(diameters_m, dtype=float)
    if d.ndim != 1:
        raise ValueError("diameters_m must be 1-D")
    if d.size and np.any(~np.isfinite(d)):
        raise ValueError("diameters_m must all be finite")
    if d.size and np.any(d <= 0):
        raise ValueError("diameters_m must be positive")
    if crater_ids is not None:
        ids = np.asarray(crater_ids)
        if ids.shape[0] != d.shape[0]:
            raise ValueError("crater_ids must have one entry per crater")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(
                "duplicate crater_ids: N must be a count of UNIQUE included "
                "craters. Deduplicate (crater.dedup) before building the table."
            )

    edges = np.asarray(edges_m, dtype=float)
    counts, n_below, n_above = bin_counts(d, edges)
    n_bins = counts.size

    low_m = edges[:-1]
    high_m = edges[1:]
    d_low_km = low_m / 1000.0
    d_high_km = high_m / 1000.0
    d_g_km = np.sqrt(d_low_km * d_high_km)
    delta_d_km = d_high_km - d_low_km

    a_km2 = np.asarray(areas_km2, dtype=float)
    if a_km2.ndim == 0:
        a_km2 = np.full(n_bins, float(a_km2))
    if a_km2.shape != (n_bins,):
        raise ValueError(
            f"areas_km2 must be scalar or have one value per bin ({n_bins}), "
            f"got shape {a_km2.shape}"
        )
    if np.any(~np.isfinite(a_km2)) or np.any(a_km2 < 0.0):
        raise ValueError("areas_km2 must be finite and non-negative")

    n_low, n_high, zero_count = poisson_interval(counts, level=confidence_level)

    has_area = a_km2 > 0.0
    # R = D_g^3 N / (A dD); the coefficient is the part independent of the count.
    coeff = np.full(n_bins, np.nan)
    np.divide(
        d_g_km**3,
        a_km2 * delta_d_km,
        out=coeff,
        where=has_area,
    )

    r = np.where(has_area & (counts > 0), coeff * counts, np.nan)
    r_low = np.where(has_area, coeff * n_low, np.nan)
    r_high = np.where(has_area, coeff * n_high, np.nan)

    flag = np.full(n_bins, FLAG_OK, dtype="<U32")
    flag[~has_area] = FLAG_NO_AREA
    flag[has_area & zero_count] = FLAG_ZERO_COUNT
    is_upper_limit = has_area & zero_count

    mc_fields: dict = {}
    if uncertainty is not None:
        draws = monte_carlo_bin_counts(d, edges, uncertainty)
        lo_p, hi_p = uncertainty.percentiles
        n_mc_low, n_mc_high = np.percentile(draws, [lo_p, hi_p], axis=0)
        mc_fields = {
            "n_mc_mean": draws.mean(axis=0),
            "n_mc_std": draws.std(axis=0, ddof=1) if uncertainty.n_draws > 1 else np.zeros(n_bins),
            "n_mc_low": n_mc_low,
            "n_mc_high": n_mc_high,
            "r_mc_low": np.where(has_area, coeff * n_mc_low, np.nan),
            "r_mc_high": np.where(has_area, coeff * n_mc_high, np.nan),
            "mc_percentiles": tuple(float(x) for x in uncertainty.percentiles),
            "mc_n_draws": int(uncertainty.n_draws),
        }

    return SFDTable(
        label=label,
        d_low_km=d_low_km,
        d_high_km=d_high_km,
        d_g_km=d_g_km,
        delta_d_km=delta_d_km,
        n=counts,
        area_km2=a_km2,
        r=r,
        r_poisson_low=r_low,
        r_poisson_high=r_high,
        n_poisson_low=n_low,
        n_poisson_high=n_high,
        is_upper_limit=is_upper_limit,
        flag=flag,
        confidence_level=float(confidence_level),
        n_below_range=n_below,
        n_above_range=n_above,
        edges_m=edges,
        **mc_fields,
    )


def require_identical_bins(*tables: SFDTable) -> np.ndarray:
    """Assert that every table shares one bin grid; return the common ``D_g``.

    Comparing two series bin by bin on an R-plot is only meaningful on identical
    bins, so this is checked rather than assumed.
    """
    if not tables:
        raise ValueError("need at least one table")
    ref = tables[0]
    for t in tables[1:]:
        if t.n_bins != ref.n_bins or not (
            np.allclose(t.d_low_km, ref.d_low_km, rtol=1e-12, atol=0.0)
            and np.allclose(t.d_high_km, ref.d_high_km, rtol=1e-12, atol=0.0)
        ):
            raise ValueError(
                f"series {t.label!r} and {ref.label!r} do not share bin edges; "
                "build both with the same log_bin_edges_m parameters"
            )
    return ref.d_g_km


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def write_bin_table_csv(tables: SFDTable | Sequence[SFDTable], path: str) -> str:
    """Write ``diameter_bins.csv``-style output; returns the path written.

    One row per bin per series, with a ``series`` column so several series share
    one file.  NaN ``R`` values are written as empty fields, which is the honest
    representation of "undefined", not 0.
    """
    if isinstance(tables, SFDTable):
        tables = [tables]
    frames = [t.to_dataframe() for t in tables]
    df = pd.concat(frames, ignore_index=True)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_diameter_bins_csv(
    tables: SFDTable | Sequence[SFDTable], out_dir: str, filename: str = "diameter_bins.csv"
) -> str:
    """Convenience wrapper writing ``<out_dir>/diameter_bins.csv``."""
    return write_bin_table_csv(tables, os.path.join(out_dir, filename))


# --------------------------------------------------------------------------- #
# The R-plot
# --------------------------------------------------------------------------- #
#: Categorical series colours, light surface (validated data-viz palette slots
#: 1, 2, 3; see the dataviz skill's references/palette.md).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
#: Marker per series: a second, non-colour channel carrying identity.
SERIES_MARKERS = ("o", "s", "^", "D")
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#dcdbd6"


@dataclass(frozen=True)
class RPlotResult:
    """What :func:`plot_rplot` produced, including the data it handed the axes.

    ``plotted`` maps each series label to the ``(x, y)`` arrays actually passed
    to the axes.  It exists so that a test can prove the axes received **raw**
    ``D_g`` and ``R`` values on logarithmic scales, and not ``log10`` of them --
    the classic double-log bug, which silently compresses a three-decade plot
    into half a decade and makes every slope wrong.
    """

    png_path: str
    svg_path: str
    figure: object
    axes: object
    plotted: dict
    upper_limits: dict


def plot_rplot(
    tables: SFDTable | Sequence[SFDTable],
    out_dir: str,
    *,
    basename: str = "rplot",
    completeness_limit_m: float | None = None,
    completeness_label: str = "completeness limit",
    title: str = "Crater relative size-frequency distribution (R-plot)",
    subtitle: str | None = None,
    show_monte_carlo: bool = True,
    ylim: tuple[float, float] | None = None,
    dpi: int = 200,
) -> RPlotResult:
    """Write ``rplot.png`` and ``rplot.svg``: ``D_g`` against ``R``.

    Both axes are set to a logarithmic **scale** and the **raw** ``D_g`` (km) and
    ``R`` values are plotted.  The data are never transformed with ``log10``
    before plotting: doing that as well as setting a log scale -- a frequent and
    hard-to-see error -- would plot ``log10(log10(R))`` and leave the tick labels
    claiming otherwise.  :class:`RPlotResult.plotted` exposes what the axes were
    given so the invariant can be tested.

    Series handling
    ---------------
    * Any number of series (catalogue vs detector, say).  All must share bin
      edges; :func:`require_identical_bins` enforces it.
    * Each series gets a colour **and** a marker shape, so identity does not rest
      on colour alone, plus a legend whenever more than one series is drawn.
    * Poisson error bars are drawn as the solid bar.  The Monte Carlo diameter
      spread, if present, is drawn as a **separate** translucent band -- never
      added to the Poisson bar.
    * ``N = 0`` bins are drawn as downward carets at their Poisson upper limit,
      annotated as upper limits.  Bins with no counting area are omitted
      entirely; they carry no information and must not be drawn at ``R = 0``.
    * ``completeness_limit_m`` marks the smallest reliably detected diameter: a
      vertical rule with the region below it shaded, because R values there are
      lower bounds, not measurements.
    * ``ylim`` fixes the ``R`` range by hand.  Left automatic, the upper limits
      of very small empty bins can stretch the axis over several decades -- which
      is honest, but a caller comparing two surveys may want a fixed range.

    Returns
    -------
    RPlotResult
    """
    # Imported lazily and without pyplot: a library must not grab a global
    # figure manager or force a GUI backend on its caller.  FigureCanvasAgg
    # renders headless; savefig switches to the SVG canvas by extension.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    if isinstance(tables, SFDTable):
        tables = [tables]
    tables = list(tables)
    if not tables:
        raise ValueError("plot_rplot needs at least one SFDTable")
    require_identical_bins(*tables)
    if len(tables) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(tables)} series exceeds the {len(SERIES_COLORS)} validated "
            "categorical colours; fold the extra series together or facet the "
            "plot rather than inventing a colour"
        )

    os.makedirs(out_dir, exist_ok=True)
    fig = Figure(figsize=(7.4, 5.2), facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_facecolor(_SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plotted: dict = {}
    upper_limits: dict = {}
    # Legend order is set deliberately: series first, then their upper limits,
    # then the measurement-error bands -- not the artist creation order.
    series_labels: list[str] = []
    limit_labels: list[str] = []
    band_labels: list[str] = []

    for i, t in enumerate(tables):
        color = SERIES_COLORS[i]
        marker = SERIES_MARKERS[i]

        good = np.isfinite(t.r) & (t.r > 0.0)
        x = t.d_g_km[good]
        y = t.r[good]
        y_lo = t.r_poisson_low[good]
        y_hi = t.r_poisson_high[good]
        # Error bars as offsets from the raw value; clipped at 0 so a bound that
        # rounds to the value itself cannot make a negative offset.
        yerr = np.vstack(
            [np.maximum(y - y_lo, 0.0), np.maximum(y_hi - y, 0.0)]
        )

        if show_monte_carlo and t.has_monte_carlo:
            band_lo = t.r_mc_low[good]
            band_hi = t.r_mc_high[good]
            band_ok = np.isfinite(band_lo) & np.isfinite(band_hi) & (band_lo > 0.0)
            if np.any(band_ok):
                order = np.argsort(x[band_ok])
                ax.fill_between(
                    x[band_ok][order],
                    band_lo[band_ok][order],
                    band_hi[band_ok][order],
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    zorder=1,
                    label=f"{t.label}: diameter-error spread",
                )
                band_labels.append(f"{t.label}: diameter-error spread")

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            color=color,
            markersize=5.5,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.0,
            elinewidth=2.0,
            capsize=0.0,
            linestyle="-",
            linewidth=1.6,
            zorder=3,
            label=t.label,
        )
        plotted[t.label] = (np.array(x, copy=True), np.array(y, copy=True))
        series_labels.append(t.label)

        ul = t.is_upper_limit & np.isfinite(t.r_poisson_high) & (t.r_poisson_high > 0.0)
        if np.any(ul):
            x_ul = t.d_g_km[ul]
            y_ul = t.r_poisson_high[ul]
            ax.errorbar(
                x_ul,
                y_ul,
                yerr=np.vstack([y_ul * 0.45, np.zeros_like(y_ul)]),
                uplims=True,
                fmt="none",
                ecolor=color,
                elinewidth=1.6,
                zorder=2,
                label=f"{t.label}: upper limit (N = 0)",
            )
            upper_limits[t.label] = (np.array(x_ul, copy=True), np.array(y_ul, copy=True))
            limit_labels.append(f"{t.label}: upper limit (N = 0)")

    if ylim is not None:
        ax.set_ylim(*ylim)

    if completeness_limit_m is not None:
        d_c_km = float(completeness_limit_m) / 1000.0
        x0, x1 = ax.get_xlim()
        ax.axvspan(min(x0, d_c_km), min(d_c_km, x1), color=_GRID, alpha=0.45, zorder=0)
        ax.axvline(d_c_km, color=_INK_SECONDARY, linestyle="--", linewidth=1.2, zorder=1)
        # The shading must not be allowed to stretch the data range.
        ax.set_xlim(x0, x1)
        ax.annotate(
            completeness_label,
            xy=(d_c_km, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=_INK_SECONDARY,
        )

    ax.set_xlabel("geometric mean diameter  $D_g$  [km]", fontsize=10, color=_INK_SECONDARY)
    ax.set_ylabel("relative density  $R = D_g^3 N / (A\\,\\Delta D)$", fontsize=10,
                  color=_INK_SECONDARY)
    ax.set_title(title, fontsize=12, color=_INK, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.015),
            xycoords="axes fraction",
            fontsize=9,
            color=_INK_SECONDARY,
            ha="left",
            va="bottom",
        )

    ax.grid(True, which="major", color=_GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.grid(True, which="minor", color=_GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9, which="both")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    wanted = [l for l in series_labels + limit_labels + band_labels if l in by_label]
    handles = [by_label[l] for l in wanted]
    labels = list(wanted)
    if len(tables) > 1 or len(handles) > 1:
        # A flat R-plot means a -3 differential slope; say so once, in the key.
        handles.append(Line2D([], [], linestyle="none"))
        labels.append("flat $R$ $\\Rightarrow$ $dN/dD \\propto D^{-3}$")
        ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=8.5,
            labelcolor=_INK_SECONDARY,
            loc="best",
        )

    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{basename}.png")
    svg_path = os.path.join(out_dir, f"{basename}.svg")
    fig.savefig(png_path, dpi=dpi, facecolor=_SURFACE)
    fig.savefig(svg_path, facecolor=_SURFACE)

    return RPlotResult(
        png_path=png_path,
        svg_path=svg_path,
        figure=fig,
        axes=ax,
        plotted=plotted,
        upper_limits=upper_limits,
    )


__all__ = [
    "BIN_INTERVAL_CONVENTION",
    "BIN_PROGRESSION",
    "DEFAULT_SEED",
    "FLAG_NO_AREA",
    "FLAG_OK",
    "FLAG_ZERO_COUNT",
    "SERIES_COLORS",
    "SERIES_MARKERS",
    "UNCERTAINTY_POLICY",
    "DiameterUncertainty",
    "RPlotResult",
    "SFDTable",
    "bin_counts",
    "bin_index",
    "log_bin_edges_m",
    "monte_carlo_bin_counts",
    "plot_rplot",
    "poisson_interval",
    "require_identical_bins",
    "size_frequency_table",
    "write_bin_table_csv",
    "write_diameter_bins_csv",
]
