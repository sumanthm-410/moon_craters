import os
os.environ.update(GDAL_HTTP_CAINFO="/root/.ccr/ca-bundle.crt",
                  CURL_CA_BUNDLE="/root/.ccr/ca-bundle.crt",
                  GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
import numpy as np, rasterio
from rasterio.windows import from_bounds
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B="/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/wms_basemaps/Moon"
SRC={"KaguyaTC":f"{B}/Kaguya_TCortho_Mosaic_Global_4096ppd.tif",
     "WAC":f"{B}/LRO_WAC_Mosaic_Global_303ppd_v3.tif",
     "LOLA":f"{B}/LRO_LOLA_DEM_Global_256ppd_v06_16bit.tif"}
CENTERS={"A_user_2.9E_85.9S":(2.9,-85.9),"B_lit_356.4E_85.8S":(-3.6,-85.8)}

fig,axes=plt.subplots(2,3,figsize=(16,8))
for r,(cname,(lon,lat)) in enumerate(CENTERS.items()):
    for c,(sname,url) in enumerate(SRC.items()):
        ax=axes[r,c]
        dlat=0.10; dlon=dlat/max(np.cos(np.radians(lat)),1e-6)  # square-ish ground
        with rasterio.open(url) as ds:
            w=from_bounds(lon-dlon,lat-dlat,lon+dlon,lat+dlat,ds.transform)
            a=ds.read(1,window=w,boundless=True,fill_value=ds.nodata or 0).astype(float)
            nod=ds.nodata
        if nod is not None: a[a==nod]=np.nan
        # squash longitude oversampling to roughly square pixels
        if a.shape[1]>4*a.shape[0]:
            f=max(1,a.shape[1]//(a.shape[0]*2)); a=a[:,::f]
        finite=a[np.isfinite(a)]
        if finite.size:
            lo,hi=np.percentile(finite,[2,98])
            ax.imshow(a,cmap="gray" if sname!="LOLA" else "terrain",vmin=lo,vmax=hi)
            ax.set_title(f"{sname} {cname}\nstretch {lo:.0f}-{hi:.0f} uniq={len(np.unique(finite))}",fontsize=8)
        ax.axis("off")
plt.suptitle("DISPLAY STRETCH ONLY - not calibrated. Malapert candidate centres",fontsize=10)
plt.tight_layout(); plt.savefig("malapert_chips.png",dpi=95,bbox_inches="tight")
print("wrote malapert_chips.png")
# numeric sanity
for sname,url in SRC.items():
    with rasterio.open(url) as ds:
        for cname,(lon,lat) in CENTERS.items():
            dlat=0.10; dlon=dlat/np.cos(np.radians(lat))
            w=from_bounds(lon-dlon,lat-dlat,lon+dlon,lat+dlat,ds.transform)
            a=ds.read(1,window=w,boundless=True,fill_value=ds.nodata or 0)
            u=np.unique(a)
            print(f"{sname:9s} {cname:22s} shape={a.shape} uniq={u.size} "
                  f"p2={np.percentile(a,2):.0f} p50={np.percentile(a,50):.0f} p98={np.percentile(a,98):.0f}")
