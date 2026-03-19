import os
import pandas as pd
from sunpy.net import Fido, attrs as a
from astropy.time import Time
import astropy.units as u
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

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

print(f"Bulk AIA 193Å hourly: {t_min.isot} → {t_max.isot}")

tag = f"{YEAR}_bulk"
outdir = os.path.join(AIA_DIR, tag)
os.makedirs(outdir, exist_ok=True)

print("\n[FETCH BULK - Resume Enabled]")
total_downloaded = 0

months = pd.date_range(t_min.datetime, t_max.datetime, freq='MS')

for i in range(len(months) - 1):

    t1 = Time(months[i])
    t2 = Time(months[i + 1])

    print(f"\nMonth {i+1}/{len(months)-1}: {t1.isot[:10]}")

    try:
        query = Fido.search(
            a.Time(t1, t2),
            a.Instrument("AIA"),
            a.Wavelength(193 * u.angstrom),
            a.Sample(1 * u.hour)
        )
    except Exception as e:
        print(f"  Search failed: {e}")
        continue

    print(f"  Found: {len(query)} files")

    if len(query) == 0:
        continue

    try:
        results = list(
            tqdm(
                Fido.fetch(
                    query,
                    path=os.path.join(outdir, "{file}"),
                    overwrite=False,        # ← critical for resume
                    max_concurrent=2
                ),
                total=len(query),
                desc="Downloading",
                leave=False
            )
        )

        # Count only successfully processed entries
        downloaded_this_month = len([r for r in results if r is not None])
        total_downloaded += downloaded_this_month

        print(f"  +{downloaded_this_month} processed")

    except Exception as e:
        print(f"  Download interrupted: {e}")
        print("  Continuing to next month...")
        continue

print(f"\n✓ Process complete")
print(f"Total files processed this run: {total_downloaded}")
print(f"Folder: {outdir}")
