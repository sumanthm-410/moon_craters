"""Tests for deduplication of crater detections.

Every scenario is synthetic with a known answer.  Detections are placed by
geodesic offset (:func:`crater.geometry.geodesic_destination`) so a stated
"40 m apart" really is 40 m of ground at 86 S, not 40 m of some planar
degree conversion.

The headline case is :func:`test_small_crater_nested_in_large_one_stays_two_craters`:
a 30 m crater centred inside a 300 m crater is two craters, and no amount of
centre proximity may merge them.
"""
import numpy as np
import pytest

from crater import dedup as dd
from crater import geometry as g

ROI_LON, ROI_LAT = 2.9, -85.9


def offset(lon_deg, lat_deg, distance_m, bearing_deg=37.0):
    """A point ``distance_m`` of ground away, by geodesic -- never by degrees."""
    return g.geodesic_destination(lon_deg, lat_deg, bearing_deg, distance_m)


def det(det_id, *, lon=ROI_LON, lat=ROI_LAT, diameter_m=100.0, confidence=0.9,
        tile_id=None, source_id=None, scale_level=None):
    return dd.Detection(det_id, lon, lat, diameter_m, confidence,
                        tile_id, source_id, scale_level)


def det_at(det_id, distance_m, *, bearing_deg=37.0, diameter_m=100.0, **kw):
    lon, lat = offset(ROI_LON, ROI_LAT, distance_m, bearing_deg)
    return det(det_id, lon=lon, lat=lat, diameter_m=diameter_m, **kw)


# --------------------------------------------------------------------- #
# Input validation and conventions
# --------------------------------------------------------------------- #
def test_detection_rejects_impossible_values():
    for kw in ({"diameter_m": 0.0}, {"diameter_m": -5.0},
               {"diameter_m": float("nan")}, {"confidence": 1.5},
               {"confidence": -0.1}, {"lat": -95.0}, {"lat": float("inf")}):
        with pytest.raises(ValueError):
            det("x", **kw)
    with pytest.raises(ValueError):
        dd.Detection("", ROI_LON, ROI_LAT, 100.0)


def test_detection_wraps_longitude_into_the_canonical_domain():
    d = det("x", lon=362.9)
    assert d.lon_deg == pytest.approx(2.9, abs=1e-9)
    assert dd.Detection("y", 180.0, ROI_LAT, 100.0).lon_deg == -180.0


def test_match_criteria_validation():
    with pytest.raises(ValueError):
        dd.MatchCriteria(centre_tol_rel=0.0)
    with pytest.raises(ValueError, match="max/min"):
        dd.MatchCriteria(diameter_ratio_max=0.9)
    with pytest.raises(ValueError, match="separation_method"):
        dd.MatchCriteria(separation_method="euclidean_pixels")


def test_duplicate_detection_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        dd.deduplicate([det("a"), det("a", diameter_m=110.0)])


def test_empty_input_gives_no_records():
    assert dd.deduplicate([]) == []


# --------------------------------------------------------------------- #
# Ground separation: projected metres, scale-aware
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("distance_m", [5.0, 50.0, 300.0, 2000.0])
def test_projected_separation_matches_the_great_circle_reference(distance_m):
    """The scale-aware projected separation must agree with the exact haversine
    distance to far better than any measurement at these ranges."""
    a = det("a")
    b = det_at("b", distance_m)
    proj = dd.ground_separation_m(a, b, method="projected")
    exact = dd.ground_separation_m(a, b, method="great_circle")
    assert proj == pytest.approx(distance_m, rel=1e-8)
    assert exact == pytest.approx(distance_m, rel=1e-8)
    assert abs(proj - exact) / exact < 1e-6


def test_raw_projected_distance_without_k_would_be_biased_high():
    """Guard on INTERFACES.md rule 2: skipping the division by k inflates every
    separation by 0.128% at 86 S, in the same direction every time."""
    a = det("a")
    b = det_at("b", 1000.0)
    xa, ya = g.forward(a.lon_deg, a.lat_deg)
    xb, yb = g.forward(b.lon_deg, b.lat_deg)
    raw = float(np.hypot(xb - xa, yb - ya))
    # k at the midpoint latitude: the projected chord is stretched by the
    # scale factor averaged along the path, which to first order is k(mid).
    k = g.point_scale_factor(0.5 * (a.lat_deg + b.lat_deg))
    assert k == pytest.approx(1.0013, abs=5e-4)
    assert raw / 1000.0 == pytest.approx(k, rel=1e-6)
    assert dd.ground_separation_m(a, b) == pytest.approx(1000.0, rel=1e-8)
    assert raw - 1000.0 > 1.0


def test_separation_is_symmetric_and_zero_for_identical_centres():
    a, b = det("a"), det_at("b", 123.0)
    for method in dd.SEPARATION_METHODS:
        assert dd.ground_separation_m(a, b, method=method) == pytest.approx(
            dd.ground_separation_m(b, a, method=method), rel=1e-12)
        assert dd.ground_separation_m(a, a, method=method) == pytest.approx(0.0, abs=1e-9)


def test_unknown_separation_method_is_rejected():
    with pytest.raises(ValueError):
        dd.ground_separation_m(det("a"), det("b", lon=3.0), method="pixels")


def test_separation_across_the_antimeridian_is_not_360_degrees():
    a = dd.Detection("a", 179.9995, ROI_LAT, 100.0)
    b = dd.Detection("b", -179.9995, ROI_LAT, 100.0)
    sep = dd.ground_separation_m(a, b)
    assert 0.0 < sep < 5.0            # ~2.1 m, not thousands of km
    assert len(dd.deduplicate([a, b])) == 1


# --------------------------------------------------------------------- #
# THE HEADLINE CASE: nested craters are distinct
# --------------------------------------------------------------------- #
def test_small_crater_nested_in_large_one_stays_two_craters():
    """A 30 m crater exactly at the centre of a 300 m crater.

    Centre separation is *zero*, so no centre criterion can separate them.  The
    diameter-ratio criterion (10.0 > 1.5) is what keeps them distinct, and the
    verdict string names the case explicitly.
    """
    big = det("big", diameter_m=300.0, confidence=0.95)
    small = det("small", diameter_m=30.0, confidence=0.80)

    decision = dd.pair_decision(big, small)
    assert decision.separation_m == pytest.approx(0.0, abs=1e-9)
    assert decision.relative_separation == pytest.approx(0.0, abs=1e-9)
    assert decision.diameter_ratio == pytest.approx(10.0, rel=1e-12)
    assert decision.merge is False
    assert decision.reason == "nested_or_overlapping_distinct_crater"

    records = dd.deduplicate([big, small])
    assert len(records) == 2
    assert {r.crater_id for r in records} == {"big", "small"}
    assert all(r.n_detections == 1 for r in records)
    assert {r.diameter_m for r in records} == {300.0, 30.0}


@pytest.mark.parametrize("offset_m", [0.0, 1.0, 10.0, 30.0, 60.0, 120.0, 149.0])
def test_nested_crater_stays_distinct_at_every_offset_inside_the_host(offset_m):
    """Sliding the 30 m crater anywhere inside the 300 m rim never merges it."""
    big = det("big", diameter_m=300.0)
    small = det_at("small", offset_m, diameter_m=30.0)
    assert len(dd.deduplicate([big, small])) == 2


@pytest.mark.parametrize("d_small,d_big", [
    (20.0, 1000.0), (20.0, 100.0), (30.0, 300.0), (50.0, 200.0),
    (100.0, 1000.0), (200.0, 400.0),
])
def test_nested_pairs_across_the_approved_size_range_stay_distinct(d_small, d_big):
    """Every nested pair whose ratio exceeds 1.5 stays two craters, across the
    whole approved 20-1000 m range (D-003)."""
    assert d_big / d_small > 1.5
    records = dd.deduplicate([det("small", diameter_m=d_small),
                              det("big", diameter_m=d_big)])
    assert len(records) == 2


def test_a_concentric_chain_of_three_nested_craters_stays_three():
    records = dd.deduplicate([det("c20", diameter_m=20.0),
                              det("c100", diameter_m=100.0),
                              det("c600", diameter_m=600.0)])
    assert len(records) == 3
    assert sorted(r.diameter_m for r in records) == [20.0, 100.0, 600.0]


def test_known_limitation_a_concentric_pair_within_the_size_ratio_is_merged():
    """Recorded limitation, not a hidden one.

    A 200 m crater concentric inside a 280 m crater has ratio 1.4 < 1.5, so
    the criteria merge it.  Centre+size alone cannot resolve concentric craters
    of similar size; that needs a rim-overlap or morphology test.  Tightening
    ``diameter_ratio_max`` to 1.2 separates this pair, at the cost of splitting
    genuine duplicates that disagree by more than 20% in size -- so the choice
    is a documented trade-off, which :func:`evaluate_dedup` is there to score.
    """
    pair = [det("outer", diameter_m=280.0, confidence=0.9),
            det("inner", diameter_m=200.0, confidence=0.9)]
    assert len(dd.deduplicate(pair)) == 1
    tighter = dd.MatchCriteria(diameter_ratio_max=1.2)
    assert len(dd.deduplicate(pair, criteria=tighter)) == 2


def test_nesting_is_not_rescued_by_a_loose_centre_tolerance():
    """Even an absurdly permissive centre tolerance cannot merge a nested pair:
    the size criterion is independent, which is the point."""
    loose = dd.MatchCriteria(centre_tol_rel=100.0)
    records = dd.deduplicate([det("big", diameter_m=300.0),
                              det("small", diameter_m=30.0)],
                             criteria=loose)
    assert len(records) == 2


# --------------------------------------------------------------------- #
# Duplicates that SHOULD merge
# --------------------------------------------------------------------- #
def test_identical_crater_in_two_overlapping_tiles_merges_to_one():
    """The same crater in the overlap of two tiles, with the small centre and
    size disagreement two independent fits would produce."""
    a = det("tileA_7", diameter_m=100.0, confidence=0.90,
            tile_id="tile_A", source_id="NAC_0001", scale_level=0)
    b_lon, b_lat = offset(ROI_LON, ROI_LAT, 3.0, bearing_deg=200.0)
    b = det("tileB_3", lon=b_lon, lat=b_lat, diameter_m=103.0, confidence=0.82,
            tile_id="tile_B", source_id="NAC_0001", scale_level=0)

    records = dd.deduplicate([a, b])
    assert len(records) == 1
    rec = records[0]
    assert rec.n_detections == 2
    assert rec.detection_ids == ("tileA_7", "tileB_3")
    assert rec.tile_ids == ("tile_A", "tile_B")
    assert rec.source_ids == ("NAC_0001",)
    assert rec.crater_id == "tileA_7"          # highest confidence
    assert rec.confidence == pytest.approx(0.90)
    assert 100.0 <= rec.diameter_m <= 103.0
    assert rec.diameter_distance_kind == g.DIAMETER_DISTANCE_KIND
    assert rec.geometry_rule == "confidence_weighted_mean"


def test_same_crater_at_two_pyramid_scales_merges():
    """A pyramid halves the resolution, so the coarse detection is noisier in
    both centre and size -- but it is the same crater."""
    fine = det("L0_11", diameter_m=240.0, confidence=0.93,
               tile_id="tile_A", source_id="NAC_0001", scale_level=0)
    c_lon, c_lat = offset(ROI_LON, ROI_LAT, 18.0, bearing_deg=315.0)
    coarse = det("L2_4", lon=c_lon, lat=c_lat, diameter_m=270.0, confidence=0.71,
                 tile_id="tile_A", source_id="NAC_0001", scale_level=2)

    records = dd.deduplicate([fine, coarse])
    assert len(records) == 1
    rec = records[0]
    assert rec.n_detections == 2
    assert rec.scale_levels == (0, 2)
    assert rec.crater_id == "L0_11"
    assert 240.0 <= rec.diameter_m <= 270.0


def test_repeated_acquisitions_of_one_crater_merge_into_one_record():
    """Three acquisitions of the same ground, each with its own source_id."""
    dets = []
    for i, (dist, diam, conf) in enumerate([(0.0, 150.0, 0.88),
                                            (6.0, 156.0, 0.91),
                                            (9.0, 144.0, 0.65)]):
        lon, lat = offset(ROI_LON, ROI_LAT, dist, bearing_deg=20.0 + 90.0 * i)
        dets.append(det(f"acq{i}", lon=lon, lat=lat, diameter_m=diam,
                        confidence=conf, tile_id=f"tile_{i}",
                        source_id=f"NAC_{i:04d}"))
    records = dd.deduplicate(dets)
    assert len(records) == 1
    assert records[0].n_detections == 3
    assert records[0].source_ids == ("NAC_0000", "NAC_0001", "NAC_0002")
    assert records[0].crater_id == "acq1"      # highest confidence 0.91


# --------------------------------------------------------------------- #
# Distinct craters that SHOULD NOT merge
# --------------------------------------------------------------------- #
def test_two_equal_craters_just_far_enough_apart_stay_distinct():
    """D = 100 m each, centre tolerance 0.25 -> the boundary is 25 m.
    24 m apart merges; 26 m apart does not.  The criterion is sharp and is
    expressed in crater diameters, not pixels."""
    near = dd.deduplicate([det("a"), det_at("b", 24.0, bearing_deg=90.0)])
    assert len(near) == 1 and near[0].n_detections == 2

    far = dd.deduplicate([det("a"), det_at("b", 26.0, bearing_deg=90.0)])
    assert len(far) == 2
    decision = dd.pair_decision(det("a"), det_at("b", 26.0, bearing_deg=90.0))
    assert decision.reason == "centre_too_far"
    assert decision.relative_separation == pytest.approx(0.26, rel=1e-6)


def test_the_criterion_scales_with_crater_size_not_with_pixels():
    """One absolute separation, two sizes, opposite verdicts.  A fixed
    threshold of, say, 25 m would get exactly one of these two cases right."""
    sep = 40.0
    small = dd.deduplicate([det("s1", diameter_m=20.0),
                            det_at("s2", sep, diameter_m=20.0)])
    assert len(small) == 2          # 40 m is two diameters apart

    large = dd.deduplicate([det("l1", diameter_m=1000.0),
                            det_at("l2", sep, diameter_m=1000.0)])
    assert len(large) == 1          # 40 m is 4% of a diameter
    assert large[0].n_detections == 2


def test_a_field_of_well_separated_equal_craters_is_not_collapsed():
    dets = [det_at(f"c{i}", 400.0 * i, bearing_deg=90.0, diameter_m=100.0)
            for i in range(6)]
    assert len(dd.deduplicate(dets)) == 6


def test_diameter_ratio_boundary_is_explicit():
    """At the default 1.5 the boundary is exact and testable."""
    co_located = {"lon": ROI_LON, "lat": ROI_LAT}
    assert dd.pair_decision(det("a", diameter_m=100.0, **co_located),
                            det("b", diameter_m=149.0, **co_located)).merge
    assert not dd.pair_decision(det("a", diameter_m=100.0, **co_located),
                                det("b", diameter_m=151.0, **co_located)).merge
    strict = dd.MatchCriteria(diameter_ratio_max=1.05)
    assert not dd.pair_decision(det("a", diameter_m=100.0, **co_located),
                                det("b", diameter_m=120.0, **co_located),
                                strict).merge


def test_pair_decision_reason_when_both_criteria_fail():
    far_small = det_at("b", 500.0, diameter_m=20.0)
    decision = dd.pair_decision(det("a", diameter_m=600.0), far_small)
    assert decision.reason == "different_size_and_place"
    assert not decision.merge


# --------------------------------------------------------------------- #
# Transitivity
# --------------------------------------------------------------------- #
def test_three_way_transitive_overlap_merges_into_one_crater():
    """A, B, C in three overlapping tiles: A-B and B-C match directly, A-C does
    not (0.40 > 0.25).  Single linkage must still yield ONE crater."""
    a = det("A", diameter_m=100.0, confidence=0.70, tile_id="tA", source_id="s1")
    b = det_at("B", 20.0, bearing_deg=90.0, diameter_m=100.0, confidence=0.95,
               tile_id="tB", source_id="s2")
    c = det_at("C", 40.0, bearing_deg=90.0, diameter_m=100.0, confidence=0.60,
               tile_id="tC", source_id="s3")

    assert dd.pair_decision(a, b).merge
    assert dd.pair_decision(b, c).merge
    ac = dd.pair_decision(a, c)
    assert not ac.merge and ac.reason == "centre_too_far"

    records = dd.deduplicate([a, b, c])
    assert len(records) == 1
    rec = records[0]
    assert rec.n_detections == 3
    assert rec.detection_ids == ("A", "B", "C")
    assert rec.tile_ids == ("tA", "tB", "tC")
    assert rec.source_ids == ("s1", "s2", "s3")
    assert rec.crater_id == "B"                 # highest confidence
    # The merged centre lies between A and C, as a mean must.
    sep_to_a = dd.ground_separation_m(a, det("m", lon=rec.lon_deg, lat=rec.lat_deg))
    assert 0.0 < sep_to_a < 40.0


def test_transitive_chain_does_not_leak_across_a_size_break():
    """A chain of equal-size duplicates may merge, but a nested crater sitting
    on top of the chain must not be dragged in by transitivity."""
    a = det("A", diameter_m=100.0)
    b = det_at("B", 20.0, bearing_deg=90.0, diameter_m=100.0)
    nested = det_at("N", 20.0, bearing_deg=90.0, diameter_m=20.0)
    records = dd.deduplicate([a, b, nested])
    assert len(records) == 2
    by_id = {r.crater_id: r for r in records}
    assert by_id["N"].n_detections == 1
    assert set(by_id) == {"A", "N"} or set(by_id) == {"B", "N"}
    merged = next(r for r in records if r.crater_id != "N")
    assert merged.detection_ids == ("A", "B")


def test_output_is_independent_of_input_order():
    rng = np.random.default_rng(20261001)
    dets = [det("A", diameter_m=100.0, confidence=0.7),
            det_at("B", 20.0, bearing_deg=90.0, diameter_m=100.0, confidence=0.95),
            det_at("C", 40.0, bearing_deg=90.0, diameter_m=100.0, confidence=0.6),
            det("N", diameter_m=20.0, confidence=0.5),
            det_at("Z", 900.0, bearing_deg=10.0, diameter_m=300.0, confidence=0.8)]
    reference = dd.deduplicate(dets)
    for _ in range(5):
        shuffled = list(dets)
        rng.shuffle(shuffled)
        got = dd.deduplicate(shuffled)
        assert [r.crater_id for r in got] == [r.crater_id for r in reference]
        for r1, r2 in zip(got, reference):
            assert r1.detection_ids == r2.detection_ids
            assert r1.diameter_m == pytest.approx(r2.diameter_m, rel=1e-12)
            assert r1.lon_deg == pytest.approx(r2.lon_deg, rel=1e-12)
            assert r1.lat_deg == pytest.approx(r2.lat_deg, rel=1e-12)


# --------------------------------------------------------------------- #
# Merge rule: surviving geometry
# --------------------------------------------------------------------- #
def test_confidence_weighted_mean_diameter_is_the_documented_weighted_average():
    a = det("a", diameter_m=100.0, confidence=0.9)
    b = det_at("b", 5.0, diameter_m=200.0, confidence=0.1)
    # 200/100 = 2.0 > 1.5 would block the merge, so widen the size criterion
    # deliberately for this arithmetic check.
    rec = dd.deduplicate([a, b], criteria=dd.MatchCriteria(diameter_ratio_max=3.0))[0]
    expected = (0.9 * 100.0 + 0.1 * 200.0) / (0.9 + 0.1)
    assert rec.diameter_m == pytest.approx(expected, rel=1e-12)
    assert rec.confidence == pytest.approx(0.9)      # max, not a sum
    # The merged centre sits near the heavier detection.
    d_to_a = dd.ground_separation_m(a, det("m", lon=rec.lon_deg, lat=rec.lat_deg))
    d_to_b = dd.ground_separation_m(b, det("m", lon=rec.lon_deg, lat=rec.lat_deg))
    assert d_to_a < d_to_b
    assert d_to_a == pytest.approx(0.1 * 5.0, abs=0.05)


def test_median_rule_resists_one_bad_detection():
    dets = [det("a", diameter_m=100.0, confidence=0.9),
            det_at("b", 4.0, diameter_m=104.0, confidence=0.9),
            det_at("c", 8.0, diameter_m=132.0, confidence=0.9)]
    med = dd.deduplicate(dets, geometry_rule="median")[0]
    mean = dd.deduplicate(dets, geometry_rule="confidence_weighted_mean")[0]
    assert med.diameter_m == pytest.approx(104.0, rel=1e-12)
    assert mean.diameter_m == pytest.approx(112.0, rel=1e-12)
    assert med.geometry_rule == "median"


def test_highest_confidence_rule_keeps_one_detection_untouched():
    a = det("a", diameter_m=100.0, confidence=0.6)
    b = det_at("b", 5.0, diameter_m=110.0, confidence=0.95)
    rec = dd.deduplicate([a, b], geometry_rule="highest_confidence")[0]
    assert rec.crater_id == "b"
    assert rec.diameter_m == 110.0
    assert (rec.lon_deg, rec.lat_deg) == pytest.approx((b.lon_deg, b.lat_deg), abs=1e-12)


def test_zero_confidence_members_fall_back_to_an_unweighted_mean():
    a = det("a", diameter_m=100.0, confidence=0.0)
    b = det_at("b", 5.0, diameter_m=110.0, confidence=0.0)
    rec = dd.deduplicate([a, b])[0]
    assert rec.diameter_m == pytest.approx(105.0, rel=1e-12)
    assert rec.confidence == 0.0


def test_unknown_geometry_rule_is_rejected():
    with pytest.raises(ValueError, match="geometry_rule"):
        dd.deduplicate([det("a")], geometry_rule="mode")


def test_merged_centre_is_averaged_in_projected_metres_across_the_antimeridian():
    """Averaging raw longitudes would put the merged centre on the far side of
    the Moon; averaging in projected metres does not."""
    a = dd.Detection("a", 179.999, ROI_LAT, 100.0, 0.9)
    b = dd.Detection("b", -179.999, ROI_LAT, 100.0, 0.9)
    rec = dd.deduplicate([a, b])[0]
    assert abs(rec.lon_deg) > 179.99
    for src in (a, b):
        sep = dd.ground_separation_m(
            src, dd.Detection("m", rec.lon_deg, rec.lat_deg, 100.0))
        assert sep < 5.0


def test_single_detection_passes_through_with_its_own_geometry():
    rec = dd.deduplicate([det("solo", diameter_m=137.0, tile_id="tA",
                              source_id="s1", scale_level=1)])[0]
    assert rec.n_detections == 1
    assert rec.crater_id == "solo"
    assert rec.diameter_m == pytest.approx(137.0, rel=1e-9)
    assert rec.lon_deg == pytest.approx(ROI_LON, abs=1e-9)
    assert rec.lat_deg == pytest.approx(ROI_LAT, abs=1e-9)
    assert rec.tile_ids == ("tA",) and rec.source_ids == ("s1",)
    assert rec.scale_levels == (1,)


# --------------------------------------------------------------------- #
# Evaluation against a reviewed sample
# --------------------------------------------------------------------- #
def _reviewed_scenario():
    """A scenario with one of each outcome, by construction.

    * ``dupA``/``dupB``  -- one crater in two tiles, reviewer says same -> TP
    * ``big``/``small``  -- nested pair, reviewer says different       -> TN
    * ``grow1``/``grow2``-- one crater measured 100 m and 400 m (a bad fit on
      one acquisition); the reviewer knows it is the same crater but the
      size criterion splits it                                        -> FN
    * ``twinA``/``twinB``-- two genuinely distinct 100 m craters 8 m apart,
      far too close for the geometry to separate                      -> FP
    """
    dupA = det("dupA", diameter_m=100.0, confidence=0.9, tile_id="tA")
    lon, lat = offset(ROI_LON, ROI_LAT, 4.0, bearing_deg=120.0)
    dupB = det("dupB", lon=lon, lat=lat, diameter_m=102.0, confidence=0.8,
               tile_id="tB")

    far = 5000.0
    blon, blat = offset(ROI_LON, ROI_LAT, far, bearing_deg=0.0)
    big = det("big", lon=blon, lat=blat, diameter_m=300.0, confidence=0.9)
    small = det("small", lon=blon, lat=blat, diameter_m=30.0, confidence=0.7)

    glon, glat = offset(ROI_LON, ROI_LAT, far, bearing_deg=90.0)
    grow1 = det("grow1", lon=glon, lat=glat, diameter_m=100.0, confidence=0.9)
    grow2 = det("grow2", lon=glon, lat=glat, diameter_m=400.0, confidence=0.6)

    tlon, tlat = offset(ROI_LON, ROI_LAT, far, bearing_deg=180.0)
    twinA = det("twinA", lon=tlon, lat=tlat, diameter_m=100.0, confidence=0.9)
    t2lon, t2lat = offset(tlon, tlat, 8.0, bearing_deg=45.0)
    twinB = det("twinB", lon=t2lon, lat=t2lat, diameter_m=100.0, confidence=0.9)

    dets = [dupA, dupB, big, small, grow1, grow2, twinA, twinB]
    reviewed = [("dupA", "dupB", True),
                ("big", "small", False),
                ("grow1", "grow2", True),
                ("twinA", "twinB", False)]
    return dets, reviewed


def test_evaluate_dedup_counts_every_outcome():
    dets, reviewed = _reviewed_scenario()
    records = dd.deduplicate(dets)
    ev = dd.evaluate_dedup(records, reviewed)
    assert ev.n_reviewed_pairs == 4
    assert ev.correct_merges == 1        # dupA + dupB
    assert ev.correct_separations == 1   # big vs small
    assert ev.missed_merges == 1         # grow1 vs grow2
    assert ev.wrong_merges == 1          # twinA + twinB
    assert ev.precision == pytest.approx(0.5)
    assert ev.recall == pytest.approx(0.5)
    assert ev.unreviewed_detection_ids == ()


def test_evaluate_dedup_perfect_and_degenerate_cases():
    dets = [det("a", diameter_m=100.0),
            det_at("b", 3.0, diameter_m=101.0),
            det("n", diameter_m=20.0)]
    records = dd.deduplicate(dets)
    perfect = dd.evaluate_dedup(records, [("a", "b", True), ("a", "n", False)])
    assert (perfect.correct_merges, perfect.wrong_merges,
            perfect.missed_merges, perfect.correct_separations) == (1, 0, 0, 1)
    assert perfect.precision == 1.0 and perfect.recall == 1.0

    # No reviewed merge at all: precision/recall are undefined, not fabricated.
    none_merged = dd.evaluate_dedup(records, [("a", "n", False)])
    assert none_merged.precision is None and none_merged.recall is None
    assert none_merged.unreviewed_detection_ids == ("b",)

    empty = dd.evaluate_dedup(records, [])
    assert empty.n_reviewed_pairs == 0
    assert empty.precision is None and empty.recall is None
    assert empty.unreviewed_detection_ids == ("a", "b", "n")


def test_evaluate_dedup_rejects_unknown_or_self_pairs():
    records = dd.deduplicate([det("a"), det("n", diameter_m=20.0)])
    with pytest.raises(ValueError, match="not present"):
        dd.evaluate_dedup(records, [("a", "ghost", True)])
    with pytest.raises(ValueError, match="twice"):
        dd.evaluate_dedup(records, [("a", "a", True)])


def test_evaluate_dedup_detects_malformed_records():
    rec = dd.deduplicate([det("a"), det_at("b", 3.0)])[0]
    with pytest.raises(ValueError, match="more than one record"):
        dd.evaluate_dedup([rec, rec], [("a", "b", True)])


def test_tightening_the_size_criterion_trades_wrong_merges_for_missed_ones():
    """The evaluator is useful only if it responds to the criteria; a stricter
    centre tolerance must remove the wrong merge and keep the missed one."""
    dets, reviewed = _reviewed_scenario()
    strict = dd.MatchCriteria(centre_tol_rel=0.05)
    ev = dd.evaluate_dedup(dd.deduplicate(dets, criteria=strict), reviewed)
    assert ev.wrong_merges == 0
    assert ev.correct_separations == 2
    assert ev.correct_merges == 1       # dupA/dupB are only 4 m apart
    assert ev.missed_merges == 1
    assert ev.precision == 1.0
