import os, time
os.environ.setdefault("GDAL_HTTP_CAINFO", "/root/.ccr/ca-bundle.crt")
os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.vrt,.jp2,.ovr")
import rasterio
from rasterio.windows import from_bounds

B = "/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/wms_basemaps/Moon"
targets = [
    ("KaguyaTC_4096ppd", f"{B}/Kaguya_TCortho_Mosaic_Global_4096ppd.tif"),
    ("LOLA_DEM_256ppd",  f"{B}/LRO_LOLA_DEM_Global_256ppd_v06_16bit.tif"),
    ("WAC_303ppd_v3",    f"{B}/LRO_WAC_Mosaic_Global_303ppd_v3.tif"),
]
for name, url in targets:
    print("="*70); print(name)
    t0=time.time()
    try:
        with rasterio.open(url) as ds:
            print(f"  open OK in {time.time()-t0:.1f}s")
            print(f"  size      : {ds.width} x {ds.height}  bands={ds.count} dtype={ds.dtypes[0]}")
            print(f"  crs       : {ds.crs}")
            print(f"  transform : {ds.transform}")
            print(f"  bounds    : {ds.bounds}")
            print(f"  nodata    : {ds.nodata}")
            print(f"  blocks    : {ds.block_shapes[:1]}  tiled={ds.profile.get('tiled')}")
            print(f"  overviews : {ds.overviews(1)[:8]}")
            # windowed read around Malapert: lon 2.9E, lat 85.9S  (deg CRS)
            for (lon, lat, tag) in [(2.9, -85.9, "user_85.9S_2.9E"), (-3.6, -85.8, "lit_85.8S_356.4E")]:
                dlon, dlat = 0.6, 0.08
                w = from_bounds(lon-dlon, lat-dlat, lon+dlon, lat+dlat, ds.transform)
                t1=time.time()
                a = ds.read(1, window=w, boundless=True, fill_value=0)
                import numpy as np
                nz = int((a!=0).sum())
                print(f"  [{tag}] window {a.shape} read {time.time()-t1:.1f}s "
                      f"min={a.min()} max={a.max()} mean={a.mean():.1f} nonzero={100*nz/a.size:.1f}%")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
