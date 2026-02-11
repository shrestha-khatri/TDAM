import os
import pandas as pd
from sunpy.net import Fido, attrs as a
from astropy.time import Time
import astropy.units as u
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm  

YEAR = 2017
DATADIR = "data"
AIA_DIR = os.path.join(DATADIR, "aia_193", str(YEAR))
OMNICSV = os.path.join(DATADIR, f"omni2_{YEAR}.csv")

os.makedirs(AIA_DIR, exist_ok=True)

print("Loading OMNI timestamps...")
omni = pd.read_csv(OMNICSV, header=None)
omni.iloc[:, 0] = pd.to_datetime(omni.iloc[:, 0])
timeindex = omni.iloc[:, 0]

print(f"OMNI loaded: {len(timeindex)}")

t_min = Time(timeindex.min() - pd.Timedelta(hours=106))
t_max = Time(timeindex.max() - pd.Timedelta(hours=86))

print(f"Bulk AIA 193Å hourly: {t_min.isot} → {t_max.isot} (~8760 files)")

tag = f"{YEAR}_bulk"
outdir = os.path.join(AIA_DIR, tag)
os.makedirs(outdir, exist_ok=True)

if len(os.listdir(outdir)) == 0:
    print("[FETCH BULK - No Repeats]")
    total_files = 0
    months = pd.date_range(t_min.datetime, t_max.datetime, freq='MS')
    
    for i in range(len(months)-1):
        t1 = Time(months[i])
        t2 = Time(months[i+1])
        print(f"\nMonth {i+1}/{len(months)-1}: {t1.isot[:10]}")
        
        query = Fido.search(
            a.Time(t1, t2),
            a.Instrument("AIA"),
            a.Wavelength(193 * u.angstrom),
            a.Sample(1 * u.hour)
        )
        print(f"  Found: {len(query)} files")
        
        if len(query) > 0:
            files = list(tqdm(
                Fido.fetch(query, path=os.path.join(outdir, "{file}"), max_concurrent=2),
                total=len(query), desc="Download", leave=False
            ))
            total_files += len(files)
            print(f"  +{len(files)} (total: {total_files})")
    
    print(f"\n✓ Bulk complete: {total_files} unique images")
else:
    print(f"[SKIP] {outdir} exists")

print(f"\nFolder: {outdir}")
print("Next: Per-t coronal hole sums from bulk[file:16][memory:1]")
