#!/usr/bin/env python3
"""Mosaic / survey / split QA figure."""
import sys, json
from pathlib import Path
import numpy as np, rasterio
from affine import Affine
from shapely.geometry import box
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from crater import splits, tiling, geometry as g

X0,Y1,N,PIX=-11000.0,132000.0,21000,1.0
X1,Y0=X0+N*PIX, Y1-N*PIX
A=ROOT/"artifacts"
fig,ax=plt.subplots(2,2,figsize=(15,15.5))

# 1. mosaic overview from the COG's overview level
with rasterio.open(A/"malapert_LO1_cog.tif") as ds:
    ov=ds.read(1,out_shape=(1050,1050))
v=ov[ov>0]; lo,hi=np.percentile(v,[1,99])
ax[0,0].imshow(ov,cmap="gray",vmin=lo,vmax=hi,extent=[X0,X1,Y0,Y1])
ax[0,0].set_title(f"LO1 controlled mosaic, 21x21 km @1 m/px (overview)\nstretch {lo:.0f}-{hi:.0f} DN — DISPLAY ONLY, uint16 source",fontsize=10)

# 2. survey mask
with rasterio.open(A/"survey_mask.tif") as ds:
    sm=ds.read(1,out_shape=(1050,1050))
im=ax[0,1].imshow(sm,cmap="viridis",vmin=0,vmax=3,extent=[X0,X1,Y0,Y1])
plt.colorbar(im,ax=ax[0,1],fraction=0.046,label="variants with valid data")
sa=json.loads((A/"survey_area.json").read_text())
ax[0,1].set_title(f"Survey coverage (LO1+LOA+LOH)\n"
  f"any {sa['any_variant']['true_surface_km2']:.1f} | >=2 {sa['two_or_more']['true_surface_km2']:.1f} | "
  f"all3 {sa['all_three']['true_surface_km2']:.1f} km² true surface",fontsize=10)

# 3. split map with buffers
SIDE=8600.0
val=box(X0,Y0,X0+SIDE,Y0+SIDE); test=box(X1-SIDE,Y1-SIDE,X1,Y1)
train=box(X0,Y0,X1,Y1).difference(val).difference(test)
regs=[splits.SplitRegion("train",train),splits.SplitRegion("val",val),splits.SplitRegion("test",test)]
tr=Affine.translation(X0,Y1)@Affine.scale(PIX,-PIX)
grid=tiling.TileGrid(tile_px=512,margin_px=64,transform=tr,pixel_scale_m=PIX,level=0)
tiles=list(tiling.iter_tiles(grid,width=N,height=N))
asg=splits.assign_tiles_to_splits(tiles,regs,buffer_m=1000.0)
COL={"train":"#4C78A8","val":"#F58518","test":"#54A24B",None:"#D0D0D0"}
cnt={}
for a_ in asg.assignments:
    f=a_.footprint
    if f is None: continue
    x,y=f.bounds[0],f.bounds[1]
    ax[1,0].add_patch(Rectangle((x,y),f.bounds[2]-x,f.bounds[3]-y,
        facecolor=COL.get(a_.split,"#D0D0D0"),edgecolor="none",alpha=0.85 if a_.split else 0.5))
    cnt[a_.split]=cnt.get(a_.split,0)+1
for nme,geo,c in [("val",val,"#F58518"),("test",test,"#54A24B")]:
    b=geo.bounds; ax[1,0].add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=c,lw=2.5))
ax[1,0].set_xlim(X0,X1); ax[1,0].set_ylim(Y0,Y1); ax[1,0].set_aspect("equal")
ax[1,0].set_title(f"Spatial splits: 8.6 km corner blocks, 1000 m ground buffer\n"
  f"train {cnt.get('train',0)} | val {cnt.get('val',0)} | test {cnt.get('test',0)} | "
  f"discarded {cnt.get(None,0)} (grey)",fontsize=10)
ax[1,0].set_xlabel("projected x (m)"); ax[1,0].set_ylabel("projected y (m)")

# 4. native-resolution crop
cx,cy=g.forward(2.9,-85.9); C=900
with rasterio.open(A/"malapert_LO1_cog.tif") as ds:
    col,row=~ds.transform*(cx,cy)
    crop=ds.read(1,window=rasterio.windows.Window(int(col-C//2),int(row-C//2),C,C))
vc=crop[crop>0]; lo2,hi2=np.percentile(vc,[1,99])
ax[1,1].imshow(crop,cmap="gray",vmin=lo2,vmax=hi2)
ax[1,1].set_title(f"Native 1 m/px crop at 2.9°E 85.9°S ({C} m across)\n"
                  f"16-bit source, stretch {lo2:.0f}-{hi2:.0f} DN",fontsize=10)
ax[1,1].axis("off")
plt.suptitle("Malapert Massif survey QA — controlled NAC mosaic, south polar stereographic, Moon 2000 sphere",fontsize=13)
plt.tight_layout(); plt.savefig(ROOT/"reports/evidence/survey_qa.png",dpi=95,bbox_inches="tight")
print("wrote reports/evidence/survey_qa.png")
print("split counts:",cnt)
