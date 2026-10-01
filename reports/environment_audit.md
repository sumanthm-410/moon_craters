# Environment Audit — verified 2026-10-01

Status key: VERIFIED = directly observed in this session.

## Host
| Item | Value | Status |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 6.18.44 | VERIFIED |
| CPU | 4 x Intel Xeon @ 2.10 GHz | VERIFIED |
| RAM | 15 GiB | VERIFIED |
| Disk | 30 GiB available on / | VERIFIED |
| GPU | none (`nvidia-smi` absent) | VERIFIED |
| Python | 3.11.15 | VERIFIED |

## Toolchain
Installed during this session from PyPI (reachable):
rasterio 1.4.4 (GDAL 3.10.3), pyproj 3.7.2, shapely 2.1.2, numpy 2.4.6,
pandas 3.0.6, scipy 1.17.1, matplotlib 3.11.2, pillow 12.3.0.

Absent: ISIS, conda/mamba, gdal CLI, ultralytics, torch, kaggle CLI.
`docker` CLI present but the **daemon is not running** and the socket does
not exist (VERIFIED 2026-10-01). No conda/mamba/apptainer either. There is
thus no mechanism available to install or run ISIS in this container.

## Network egress policy (the governing constraint)
Outbound HTTPS passes a policy proxy. Denials are `CONNECT ... 403` and are
policy decisions, not transient errors.

### DENIED (403) — VERIFIED
pds.lroc.asu.edu, wms.lroc.asu.edu, quickmap.lroc.asu.edu,
lroc.sese.asu.edu, www.lroc.asu.edu, lroc.im-ldi.com, data.lroc.im-ldi.com,
ode.rsl.wustl.edu, oderest.rsl.wustl.edu, pds-geosciences.wustl.edu,
pds-imaging.jpl.nasa.gov, pdsimage2.wr.usgs.gov, planetarymaps.usgs.gov,
astrogeology.usgs.gov, naif.jpl.nasa.gov, trek.nasa.gov, moon.nasa.gov,
www.kaggle.com, zenodo.org, figshare.com, huggingface.co, data.kitware.com

The `WebFetch` tool is subject to the same policy (EGRESS_BLOCKED on
pds.lroc.asu.edu), so it is not an alternate data path.

### ALLOWED — VERIFIED
pypi.org, files.pythonhosted.org, github.com, container registries
(registry-1.docker.io, ghcr.io, public.ecr.aws), conda.anaconda.org,
repo.anaconda.com, s3.amazonaws.com,
asc-pds-services.s3.us-west-2.amazonaws.com

## Consequence
Three independent, simultaneous blockers, all from the same cause:
1. IMAGERY  — no LROC NAC EDR/CDR/RDR source is reachable.
2. LABELS   — the Robbins catalogue host (PDS/USGS) and the common
              mirrors (Zenodo, Figshare, Kaggle) are all denied.
3. COMPUTE  — kaggle.com is denied, so Kaggle GPU training cannot run,
              and there is no local GPU.

Remedy: the environment's Network access setting (cloud environment menu in
the session title bar -> Edit) must either use a broader access level or add
the required hosts to the allowed domains.
Reference: https://code.claude.com/docs/en/claude-code-on-the-web


## Pinned environment
See `requirements.txt`. GDAL 3.10.3, PROJ 9.5.1 (via rasterio 1.4.4 /
pyproj 3.7.2), Python 3.11.15.
