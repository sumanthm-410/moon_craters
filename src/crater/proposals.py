"""Model-free crater proposals from low-Sun shadow/rim signatures.

These are ANNOTATION AIDS, not detections and not labels.  Every proposal is
emitted with ``source_label_type = "model_proposal"`` and
``review_status = "unreviewed"`` and must stay that way until a human accepts
or corrects it (docs/annotation_guide.md).  Nothing here may be promoted to
``human``, and an accepted proposal becomes
``model_proposal`` + ``accepted`` -- never ``human``.

Method
------
At the grazing solar incidence typical of 86 S a crater of diameter D shows a
characteristic antisymmetric signature: a bright lit wall on the sunward side
and a dark shadowed wall directly opposite.  A matched filter for that
signature is an oriented template, +1 over the sunward half-disc and -1 over
the anti-sunward half-disc, correlated with the image at a range of scales.

This is a classical, training-free detector.  It keys on the SAME shadow cue
that makes a shadow outline an unreliable rim (D-005), so its box is a
localisation hint only and its diameter is NOT a measurement.

Deliberate limitations, stated rather than hidden:
  * it responds to any sufficiently strong illumination gradient at the right
    scale and orientation, so boulders, scarps and slope breaks produce false
    positives;
  * it under-responds to degraded craters with no sharp rim, and to craters
    fully inside a larger shadow;
  * no recall or precision figure may be quoted for it until a human has
    reviewed a sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve

#: Fixed provenance for everything this module emits.
SOURCE_LABEL_TYPE = "model_proposal"
REVIEW_STATUS = "unreviewed"


@dataclass(frozen=True)
class Proposal:
    """One unreviewed crater candidate, in patch-pixel coordinates."""

    x_px: float
    y_px: float
    diameter_px: float
    score: float
    scale_m: float
    source_label_type: str = SOURCE_LABEL_TYPE
    review_status: str = REVIEW_STATUS

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        r = self.diameter_px / 2.0
        return (self.x_px - r, self.y_px - r, self.x_px + r, self.y_px + r)


def _oriented_template(radius_px: float, azimuth_deg: float) -> np.ndarray:
    """+1 on the sunward half-disc, -1 on the anti-sunward half-disc.

    The template is zero-mean, so a uniform region gives zero response and the
    filter measures the illumination asymmetry rather than brightness.
    """
    r = max(float(radius_px), 1.5)
    n = int(np.ceil(r)) * 2 + 1
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n] - c
    disc = (xx * xx + yy * yy) <= r * r
    # image row index increases downward, so negate the y component
    ux, uy = np.cos(np.radians(azimuth_deg)), -np.sin(np.radians(azimuth_deg))
    proj = xx * ux + yy * uy
    t = np.zeros((n, n), dtype=np.float32)
    t[disc & (proj > 0)] = 1.0
    t[disc & (proj < 0)] = -1.0
    pos, neg = (t > 0).sum(), (t < 0).sum()
    if pos and neg:                      # enforce exact zero mean
        t[t > 0] /= pos
        t[t < 0] /= neg
    return t


def estimate_solar_azimuth(image: np.ndarray, radius_px: float = 8.0,
                           n_azimuth: int = 16) -> float:
    """Estimate the illumination azimuth from the image itself.

    Returns the azimuth whose oriented template gives the strongest upper-tail
    response, i.e. the direction in which crater signatures actually line up.
    """
    img = _prepare(image)
    best_az, best_val = 0.0, -np.inf
    for az in np.linspace(0.0, 360.0, n_azimuth, endpoint=False):
        resp = fftconvolve(img, _oriented_template(radius_px, az)[::-1, ::-1], mode="same")
        val = float(np.percentile(resp, 99.5))
        if val > best_val:
            best_az, best_val = float(az), val
    return best_az


def _prepare(image: np.ndarray) -> np.ndarray:
    """Robustly standardise a patch for correlation (valid pixels only)."""
    a = image.astype(np.float32)
    valid = a > 0
    if not valid.any():
        return np.zeros_like(a)
    lo, hi = np.percentile(a[valid], [2, 98])
    if hi <= lo:
        return np.zeros_like(a)
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = float(np.median(out[valid]))
    return out


def _correlate(img: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """Correlate with reflect-padding, so tile borders do not create response.

    ``fftconvolve(..., mode="same")`` zero-pads, which manufactures a large
    artificial step at every tile edge and produced dense spurious clusters
    along the top border in the first version of this detector.
    """
    pad = tpl.shape[0] // 2 + 1
    a = np.pad(img, pad, mode="reflect")
    r = fftconvolve(a, tpl[::-1, ::-1], mode="same")
    r = r[pad:pad + img.shape[0], pad:pad + img.shape[1]]
    # suppress the remaining border band entirely: a template that does not
    # fit wholly inside the tile cannot evidence a crater.
    m = tpl.shape[0] // 2 + 1
    r[:m, :] = r[-m:, :] = r[:, :m] = r[:, -m:] = -np.inf
    return r


def _local_nms(cands: list[Proposal], centre_tol_rel: float = 0.25,
               diameter_ratio_max: float = 1.5,
               concentric_ratio_max: float = 3.0) -> list[Proposal]:
    """Suppress duplicates with the same rule crater.dedup uses.

    Two candidates merge only if their centres are within
    ``centre_tol_rel`` x mean diameter AND their diameters are within
    ``diameter_ratio_max``.  The diameter-ratio test is what keeps a small
    crater nested inside a large one as a separate candidate.
    """
    kept: list[Proposal] = []
    for c in sorted(cands, key=lambda p: -p.score):
        dup = False
        for k in kept:
            dmean = 0.5 * (c.diameter_px + k.diameter_px)
            sep = float(np.hypot(c.x_px - k.x_px, c.y_px - k.y_px))
            ratio = max(c.diameter_px, k.diameter_px) / max(min(c.diameter_px, k.diameter_px), 1e-9)
            dmin = min(c.diameter_px, k.diameter_px)
            # (a) same crater found at a similar scale
            if sep <= centre_tol_rel * dmean and ratio <= diameter_ratio_max:
                dup = True
                break
            # (b) same crater found at a DIFFERENT scale: near-concentric.
            # A genuine nested crater sits inside a much larger one and has a
            # ratio well above this; a multi-scale duplicate of one crater is
            # essentially concentric. This is a PROPOSAL-stage heuristic and is
            # deliberately looser than crater.dedup's measurement-stage rule,
            # because a redundant proposal costs a reviewer a glance whereas a
            # merged pair costs a missed crater.
            if sep <= 0.25 * dmin and ratio <= concentric_ratio_max:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def _verify(img: np.ndarray, x: int, y: int, r_px: float,
            azimuth_deg: float, min_contrast: float) -> bool:
    """Require a real dark/bright pair, not merely a gradient.

    A broad slope produces a large correlation response with no crater. This
    checks that the anti-sunward half is genuinely darker than the local
    surroundings AND the sunward half genuinely brighter, which a smooth
    regional gradient at the same scale does not satisfy as strongly.
    """
    r = int(np.ceil(r_px))
    y0, y1 = max(y - r, 0), min(y + r + 1, img.shape[0])
    x0, x1 = max(x - r, 0), min(x + r + 1, img.shape[1])
    sub = img[y0:y1, x0:x1]
    if sub.size < 9:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    yy = yy - y
    xx = xx - x
    disc = (xx * xx + yy * yy) <= r_px * r_px
    if disc.sum() < 8:
        return False
    ux, uy = np.cos(np.radians(azimuth_deg)), -np.sin(np.radians(azimuth_deg))
    proj = xx * ux + yy * uy
    lit = disc & (proj > 0)
    sha = disc & (proj < 0)
    if lit.sum() < 4 or sha.sum() < 4:
        return False
    ring = (~disc) & (xx * xx + yy * yy <= (1.6 * r_px) ** 2)
    if ring.sum() < 8:
        return False
    local = float(np.median(sub[ring]))
    return (float(np.median(sub[sha])) < local - 0.3 * min_contrast and
            float(np.median(sub[lit])) > local + 0.3 * min_contrast)


def propose(image: np.ndarray, pixel_scale_m: float, *,
            diameters_m=(20.0, 28.0, 40.0, 56.0, 80.0, 112.0, 160.0, 220.0),
            azimuth_deg: float | None = None,
            min_score: float = 0.18,
            max_per_scale: int = 400) -> list[Proposal]:
    """Propose crater candidates in one patch.

    ``image`` is raw source DN; 0 is treated as fill and excluded.
    Returns proposals sorted by descending score.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2-D")
    if pixel_scale_m <= 0:
        raise ValueError("pixel_scale_m must be positive")

    img = _prepare(image)
    if not np.any(img):
        return []
    az = estimate_solar_azimuth(image) if azimuth_deg is None else float(azimuth_deg)

    cands: list[Proposal] = []
    for d_m in diameters_m:
        r_px = 0.5 * d_m / pixel_scale_m
        if r_px < 2.0 or 2 * r_px >= min(img.shape) * 0.5:
            continue
        tpl = _oriented_template(r_px, az)
        resp = _correlate(img, tpl)

        # The template is zero-mean and half-normalised, so the response IS
        # mean(sunward half) - mean(anti-sunward half) in units of the
        # stretched image.  min_score is therefore an ABSOLUTE contrast
        # requirement, comparable across scales.  The earlier version used a
        # per-scale 99th percentile, which is relative and so fired on ~1% of
        # pixels whether or not any crater was present.
        resp = np.where(np.isfinite(resp), resp, -np.inf)
        peaks = resp >= min_score
        if not peaks.any():
            continue
        # keep only strict local maxima within one radius
        k = max(int(r_px), 1)
        mx = maximum_filter(np.where(np.isfinite(resp), resp, -1.0), size=2 * k + 1)
        peaks &= resp >= mx
        ys, xs = np.where(peaks)
        if ys.size == 0:
            continue
        order = np.argsort(-resp[ys, xs])[:max_per_scale]
        for i in order:
            y, x = int(ys[i]), int(xs[i])
            if not _verify(img, x, y, r_px, az, min_score):
                continue
            cands.append(Proposal(x_px=float(x), y_px=float(y),
                                  diameter_px=2.0 * r_px,
                                  score=float(resp[y, x]), scale_m=float(d_m)))
    return _local_nms(cands)
