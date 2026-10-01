"""Deduplication of crater detections across tiles, pyramid scales and acquisitions.

Why a fixed pixel distance is wrong
-----------------------------------
The same crater is seen more than once: in the overlap of two neighbouring
tiles, at two levels of an image pyramid, and in repeated acquisitions of the
same ground.  Collapsing those duplicates with a *fixed* threshold (say "merge
anything within 10 px") fails in both directions at once: 10 px is several
diameters for a 20 m crater, so distinct small craters get merged; and it is a
twentieth of a diameter for a 1000 m crater, so two detections of the same
large crater stay split.  Every criterion here is therefore **relative to
crater size**.

Coordinate space
----------------
Centres are compared in **projected metres** (the project's south polar
stereographic CRS), and the projected separation is divided by
:func:`crater.geometry.point_scale_factor` at the midpoint latitude to give a
*ground* separation -- INTERFACES.md rule 2.  This is the default
``method="projected"``.  A separation obtained this way is a chord in the
projected plane; for the separations that matter here (tens of metres to a few
kilometres) it agrees with the exact great-circle distance to better than
1e-6 relative, which ``tests/test_dedup.py`` verifies.  ``method="great_circle"``
asks for the exact value instead, and is the right choice if ever comparing
detections thousands of kilometres apart, where the midpoint-``k``
approximation breaks down.

All distances and diameters are planimetric great-circle lengths on the Moon
2000 sphere -- :data:`crater.geometry.DIAMETER_DISTANCE_KIND`, never
terrain-surface (INTERFACES.md rule 3).

The nested-crater rule
----------------------
A small crater sitting inside a larger one is a **different crater**.  Its
centre separation can be exactly zero, so no centre criterion of any kind can
tell it from a duplicate.  What separates the two cases is the *diameter
ratio*: two detections of one crater must agree in size, while a nested pair
differs by a large factor.  :attr:`MatchCriteria.diameter_ratio_max` is that
rule, it is explicit, and it is load-bearing -- see
``test_dedup.py::test_small_crater_nested_in_large_one_stays_two_craters``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as g
from .body import MOON, Body

__all__ = [
    "Detection",
    "MatchCriteria",
    "PairDecision",
    "CraterRecord",
    "DedupEvaluation",
    "SEPARATION_METHODS",
    "GEOMETRY_RULES",
    "ground_separation_m",
    "pair_decision",
    "deduplicate",
    "evaluate_dedup",
]

SEPARATION_METHODS = ("projected", "great_circle")

#: Documented rules for the geometry that survives a merge.  See
#: :func:`deduplicate` for the exact definitions.
GEOMETRY_RULES = ("confidence_weighted_mean", "median", "highest_confidence")

_EPS = 1e-12


def _finite(name: str, value) -> float:
    if value is None:
        raise ValueError(f"{name} is required but was None")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detection:
    """One detection of one crater, from one tile at one pyramid scale.

    Parameters
    ----------
    detection_id:
        Unique within the batch passed to :func:`deduplicate`; duplicates are
        rejected rather than silently coalesced.
    lon_deg, lat_deg:
        Centre, degrees, planetocentric, east-positive.
    diameter_m:
        Ground (planimetric) diameter in metres.  Positive and finite.
    confidence:
        Detector score in [0, 1].  Used as the merge weight and to pick the
        representative; ``0.0`` is allowed and means "no weight".
    tile_id, source_id, scale_level:
        Provenance.  ``source_id`` identifies the acquisition / image product,
        ``scale_level`` the pyramid level (0 = full resolution).  All optional,
        all carried through the merge as lists.
    """

    detection_id: str
    lon_deg: float
    lat_deg: float
    diameter_m: float
    confidence: float = 1.0
    tile_id: Optional[str] = None
    source_id: Optional[str] = None
    scale_level: Optional[int] = None

    def __post_init__(self) -> None:
        if not str(self.detection_id):
            raise ValueError("detection_id must be a non-empty string")
        lon = _finite("lon_deg", self.lon_deg)
        lat = _finite("lat_deg", self.lat_deg)
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"lat_deg must be within [-90, 90], got {lat}")
        d = _finite("diameter_m", self.diameter_m)
        if d <= 0.0:
            raise ValueError(f"diameter_m must be positive, got {d}")
        c = _finite("confidence", self.confidence)
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"confidence must be within [0, 1], got {c}")
        if self.scale_level is not None and int(self.scale_level) != self.scale_level:
            raise ValueError(f"scale_level must be an int or None, got {self.scale_level!r}")
        # Normalise longitude into the canonical domain so two detections near
        # the antimeridian cannot be compared across a 360 deg offset.
        object.__setattr__(self, "lon_deg", float(g.wrap_longitude(lon)))
        object.__setattr__(self, "lat_deg", lat)
        object.__setattr__(self, "diameter_m", d)
        object.__setattr__(self, "confidence", c)


@dataclass(frozen=True)
class MatchCriteria:
    """When two detections describe the same crater.  Both tests must pass.

    Attributes
    ----------
    centre_tol_rel:
        Maximum centre separation as a fraction of the **mean** of the two
        diameters.  The mean is used so the criterion is symmetric in the two
        detections.  Default 0.25: duplicates of one crater, whose centres are
        fitted independently in two images, are expected to agree to well
        within a quarter of a diameter; two genuinely distinct craters of
        similar size that close together would overlap so heavily that they
        could not be told apart anyway.
    diameter_ratio_max:
        Maximum ``max(D_a, D_b) / min(D_a, D_b)``.  Default 1.5: repeated
        measurements of one crater across pyramid levels and illumination
        conditions scatter by tens of percent, but not by a factor of 1.5.
        **This is the rule that keeps a nested small crater distinct from its
        host**, because the centre criterion alone cannot.
    separation_method:
        ``"projected"`` (default) or ``"great_circle"`` -- see the module
        docstring.
    """

    centre_tol_rel: float = 0.25
    diameter_ratio_max: float = 1.5
    separation_method: str = "projected"

    def __post_init__(self) -> None:
        t = _finite("centre_tol_rel", self.centre_tol_rel)
        if t <= 0.0:
            raise ValueError(f"centre_tol_rel must be positive, got {t}")
        r = _finite("diameter_ratio_max", self.diameter_ratio_max)
        if r < 1.0:
            raise ValueError(
                f"diameter_ratio_max must be >= 1 (it is max/min), got {r}")
        if self.separation_method not in SEPARATION_METHODS:
            raise ValueError(
                f"separation_method must be one of {SEPARATION_METHODS}, "
                f"got {self.separation_method!r}")


@dataclass(frozen=True)
class PairDecision:
    """Why one pair of detections was or was not merged.

    ``reason`` is a stable string:

    ``"same_crater"``
        both criteria passed.
    ``"centre_too_far"``
        the centres are too far apart relative to crater size.
    ``"nested_or_overlapping_distinct_crater"``
        the centres are close enough, but the diameters differ by more than
        ``diameter_ratio_max``.  This is the nested-crater verdict.
    ``"different_size_and_place"``
        both criteria failed.
    """

    a_id: str
    b_id: str
    separation_m: float
    relative_separation: float
    diameter_ratio: float
    merge: bool
    reason: str
    separation_method: str


# --------------------------------------------------------------------------- #
# Ground separation
# --------------------------------------------------------------------------- #
def ground_separation_m(a: Detection, b: Detection, *,
                        method: str = "projected",
                        body: Body = MOON,
                        lon_0: float = 0.0,
                        k0: float = 1.0) -> float:
    """Ground (planimetric) separation between two detection centres, metres.

    ``method="projected"``
        Project both centres, take the Euclidean distance in projected metres,
        then divide by ``point_scale_factor`` at the midpoint latitude.  This
        is the scale-aware ground distance the project asks for: the raw
        projected distance would be 0.12% too long at the ROI latitude.
    ``method="great_circle"``
        The exact haversine distance from :mod:`crater.geometry`, used as the
        reference implementation.
    """
    if method not in SEPARATION_METHODS:
        raise ValueError(f"method must be one of {SEPARATION_METHODS}, got {method!r}")
    if method == "great_circle":
        return float(g.great_circle_distance(a.lon_deg, a.lat_deg,
                                             b.lon_deg, b.lat_deg, body=body))
    xa, ya = g.forward(a.lon_deg, a.lat_deg, body=body, lon_0=lon_0, k0=k0)
    xb, yb = g.forward(b.lon_deg, b.lat_deg, body=body, lon_0=lon_0, k0=k0)
    d_projected = float(np.hypot(xb - xa, yb - ya))
    mid_lat = 0.5 * (a.lat_deg + b.lat_deg)
    k = float(g.point_scale_factor(mid_lat, k0=k0))
    if not np.isfinite(k) or k <= 0.0:  # pragma: no cover - guarded by Detection
        raise ValueError(f"unusable scale factor k={k} at midpoint latitude {mid_lat}")
    return d_projected / k


# --------------------------------------------------------------------------- #
# Pairwise decision
# --------------------------------------------------------------------------- #
def pair_decision(a: Detection, b: Detection,
                  criteria: Optional[MatchCriteria] = None, *,
                  body: Body = MOON,
                  lon_0: float = 0.0,
                  k0: float = 1.0) -> PairDecision:
    """Decide whether two detections describe the same crater."""
    criteria = criteria or MatchCriteria()
    sep = ground_separation_m(a, b, method=criteria.separation_method,
                              body=body, lon_0=lon_0, k0=k0)
    mean_d = 0.5 * (a.diameter_m + b.diameter_m)
    rel_sep = sep / mean_d  # mean_d > 0 by Detection validation
    ratio = max(a.diameter_m, b.diameter_m) / min(a.diameter_m, b.diameter_m)

    centre_ok = rel_sep <= criteria.centre_tol_rel
    size_ok = ratio <= criteria.diameter_ratio_max
    if centre_ok and size_ok:
        reason = "same_crater"
    elif centre_ok and not size_ok:
        reason = "nested_or_overlapping_distinct_crater"
    elif size_ok and not centre_ok:
        reason = "centre_too_far"
    else:
        reason = "different_size_and_place"
    return PairDecision(
        a_id=a.detection_id, b_id=b.detection_id,
        separation_m=sep, relative_separation=rel_sep, diameter_ratio=ratio,
        merge=bool(centre_ok and size_ok), reason=reason,
        separation_method=criteria.separation_method,
    )


# --------------------------------------------------------------------------- #
# Merged output
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CraterRecord:
    """One crater after deduplication.

    ``crater_id`` is stable: it is the ``detection_id`` of the cluster's
    *representative* (highest confidence, ties broken by the lexicographically
    smallest id), so the same input set in any order yields the same id.  If a
    later run adds detections and changes the cluster membership the
    representative may change, which is why ``detection_ids`` is retained --
    provenance, not the id, is the audit trail.
    """

    crater_id: str
    lon_deg: float
    lat_deg: float
    diameter_m: float
    confidence: float
    n_detections: int
    detection_ids: Tuple[str, ...]
    tile_ids: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    scale_levels: Tuple[int, ...]
    geometry_rule: str
    diameter_distance_kind: str = g.DIAMETER_DISTANCE_KIND


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        p = self._parent
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self._parent[max(ri, rj)] = min(ri, rj)


def _merge_geometry(members: Sequence[Detection], rule: str, *,
                    body: Body, lon_0: float, k0: float):
    """Return ``(lon, lat, diameter_m, confidence)`` for a merged cluster.

    Rules (all documented, all deterministic):

    ``"confidence_weighted_mean"`` (default)
        Centres are projected to metres, averaged with weights equal to
        ``confidence``, and unprojected.  Averaging in projected metres is
        mandatory near the pole: an average of raw longitudes is meaningless
        there and breaks outright across the antimeridian.  The diameter is the
        confidence-weighted mean of the member diameters.  If every weight is
        zero the plain arithmetic mean is used.
    ``"median"``
        Component-wise median of the projected x, y and of the diameter.  More
        robust to one bad detection, but for an even number of members it
        returns a midpoint rather than an observed value.
    ``"highest_confidence"``
        The representative's own geometry, untouched.  No averaging at all.

    The merged confidence is the **maximum** of the members, never a sum or a
    probabilistic combination: repeated detections of one crater in overlapping
    tiles are not independent evidence, and treating them as such would inflate
    scores for craters that happen to lie in an overlap.
    """
    conf = np.array([m.confidence for m in members], dtype=float)
    diam = np.array([m.diameter_m for m in members], dtype=float)
    x, y = g.forward(np.array([m.lon_deg for m in members], dtype=float),
                     np.array([m.lat_deg for m in members], dtype=float),
                     body=body, lon_0=lon_0, k0=k0)
    merged_conf = float(conf.max())

    if rule == "highest_confidence":
        rep = members[0]
        return rep.lon_deg, rep.lat_deg, rep.diameter_m, merged_conf
    if rule == "median":
        xm, ym = float(np.median(x)), float(np.median(y))
        lon, lat = g.inverse(xm, ym, body=body, lon_0=lon_0, k0=k0)
        return float(lon), float(lat), float(np.median(diam)), merged_conf
    if rule == "confidence_weighted_mean":
        w = conf if conf.sum() > _EPS else np.ones_like(conf)
        xm = float(np.sum(w * x) / np.sum(w))
        ym = float(np.sum(w * y) / np.sum(w))
        lon, lat = g.inverse(xm, ym, body=body, lon_0=lon_0, k0=k0)
        return float(lon), float(lat), float(np.sum(w * diam) / np.sum(w)), merged_conf
    raise ValueError(f"geometry_rule must be one of {GEOMETRY_RULES}, got {rule!r}")


def deduplicate(detections: Iterable[Detection], *,
                criteria: Optional[MatchCriteria] = None,
                geometry_rule: str = "confidence_weighted_mean",
                body: Body = MOON,
                lon_0: float = 0.0,
                k0: float = 1.0) -> List[CraterRecord]:
    """Collapse duplicate detections into one :class:`CraterRecord` per crater.

    Clustering is **single-linkage** over the pairwise decision of
    :func:`pair_decision`: if A matches B and B matches C then A, B and C form
    one crater even if A and C are not a direct match.  That transitivity is
    required -- three tiles can overlap the same crater, and the two outermost
    detections may sit just beyond the direct threshold.  The known cost is
    chaining: a dense line of similar-sized craters each within tolerance of
    its neighbour would merge into one.  With ``centre_tol_rel = 0.25`` such a
    chain requires craters overlapping by more than 75% of their diameters, so
    the risk is accepted and recorded here rather than hidden.

    The scan is O(n^2) in the number of detections, which is ample for a tile
    or an ROI batch.  A grid or KD-tree index is the production path and would
    not change any decision, only the cost.

    Returns records sorted by ``crater_id`` for reproducibility.
    """
    dets = list(detections)
    criteria = criteria or MatchCriteria()
    if geometry_rule not in GEOMETRY_RULES:
        raise ValueError(f"geometry_rule must be one of {GEOMETRY_RULES}, got {geometry_rule!r}")
    ids = [d.detection_id for d in dets]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"detection_id values must be unique; repeated: {dupes}")
    if not dets:
        return []

    uf = _UnionFind(len(dets))
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if pair_decision(dets[i], dets[j], criteria,
                             body=body, lon_0=lon_0, k0=k0).merge:
                uf.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(dets)):
        clusters.setdefault(uf.find(i), []).append(i)

    records: List[CraterRecord] = []
    for idxs in clusters.values():
        members = [dets[i] for i in idxs]
        # Representative: highest confidence, ties -> smallest detection_id.
        members.sort(key=lambda d: (-d.confidence, d.detection_id))
        lon, lat, diam, conf = _merge_geometry(
            members, geometry_rule, body=body, lon_0=lon_0, k0=k0)
        records.append(CraterRecord(
            crater_id=members[0].detection_id,
            lon_deg=float(g.wrap_longitude(lon)),
            lat_deg=float(lat),
            diameter_m=float(diam),
            confidence=float(conf),
            n_detections=len(members),
            detection_ids=tuple(sorted(m.detection_id for m in members)),
            tile_ids=tuple(sorted({m.tile_id for m in members if m.tile_id is not None})),
            source_ids=tuple(sorted({m.source_id for m in members if m.source_id is not None})),
            scale_levels=tuple(sorted({m.scale_level for m in members
                                       if m.scale_level is not None})),
            geometry_rule=geometry_rule,
        ))
    records.sort(key=lambda r: r.crater_id)
    return records


# --------------------------------------------------------------------------- #
# Evaluation against a reviewed sample
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DedupEvaluation:
    """Deduplication scored against human-reviewed pairs.

    The unit is a *reviewed pair of detections*, each labelled by a reviewer as
    the same crater or not.

    * ``correct_merges``      -- labelled same, ended in one cluster (TP)
    * ``wrong_merges``        -- labelled different, ended in one cluster (FP)
    * ``missed_merges``       -- labelled same, left in different clusters (FN)
    * ``correct_separations`` -- labelled different, left separate (TN)

    ``precision`` and ``recall`` are over merge decisions and are ``None``
    rather than a fabricated 0 or 1 when their denominator is zero.
    ``unreviewed_detection_ids`` lists ids present in the records but absent
    from the reviewed sample, so a claim of "evaluated" cannot hide how little
    of the output was actually checked.
    """

    correct_merges: int
    wrong_merges: int
    missed_merges: int
    correct_separations: int
    n_reviewed_pairs: int
    precision: Optional[float]
    recall: Optional[float]
    unreviewed_detection_ids: Tuple[str, ...]


def evaluate_dedup(records: Sequence[CraterRecord],
                   reviewed_pairs: Iterable[Tuple[str, str, bool]],
                   ) -> DedupEvaluation:
    """Score ``records`` against reviewed detection pairs.

    Parameters
    ----------
    records:
        Output of :func:`deduplicate`.
    reviewed_pairs:
        Iterable of ``(detection_id_a, detection_id_b, same_crater)``.
        ``same_crater`` is the reviewer's verdict.  Both ids must appear in
        ``records``; an unknown id raises rather than being scored as a miss,
        because silently ignoring it would flatter the result.
    """
    cluster_of: Dict[str, str] = {}
    for rec in records:
        for det_id in rec.detection_ids:
            if det_id in cluster_of:
                raise ValueError(
                    f"detection {det_id!r} appears in more than one record; "
                    "deduplicate() output is malformed")
            cluster_of[det_id] = rec.crater_id

    tp = fp = fn = tn = 0
    reviewed_ids = set()
    n_pairs = 0
    for pair in reviewed_pairs:
        a_id, b_id, same = pair
        for det_id in (a_id, b_id):
            if det_id not in cluster_of:
                raise ValueError(
                    f"reviewed detection {det_id!r} is not present in the "
                    "deduplicated records")
        if a_id == b_id:
            raise ValueError(f"reviewed pair references one detection twice: {a_id!r}")
        reviewed_ids.update((a_id, b_id))
        n_pairs += 1
        merged = cluster_of[a_id] == cluster_of[b_id]
        if same and merged:
            tp += 1
        elif same and not merged:
            fn += 1
        elif (not same) and merged:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return DedupEvaluation(
        correct_merges=tp,
        wrong_merges=fp,
        missed_merges=fn,
        correct_separations=tn,
        n_reviewed_pairs=n_pairs,
        precision=precision,
        recall=recall,
        unreviewed_detection_ids=tuple(sorted(set(cluster_of) - reviewed_ids)),
    )
