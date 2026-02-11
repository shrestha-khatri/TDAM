import os
import io
import requests
import pandas as pd
import warnings
from sunpy.net import Fido, attrs as a
from astropy.time import Time

warnings.filterwarnings("ignore")

YEAR = "2017"
DATADIR = "data"
SDO_CSV = f"sdo_aia131_{YEAR}.csv"

os.makedirs(DATADIR, exist_ok=True)

print(f"Fetching SDO AIA 131Å images for {YEAR}...")

OMNI_HAPI_URL = (
    f"https://cdaweb.gsfc.nasa.gov/hapi/data?"
    f"id=OMNI2_V1|parameters=Hour|start={YEAR}-01-01T00:00:00Z|"
    f"end={YEAR}-12-31T23:59:59Z&format=csv"
)

try:
    r = requests.get(OMNI_HAPI_URL, timeout=120)
    r.raise_for_status()
    omni_df = pd.read_csv(io.StringIO(r.text.strip()))
    timeindex = pd.to_datetime(omni_df.iloc[:, 0])
    print(f"OMNI loaded: {len(timeindex)} points")
except Exception as e:
    print(f"Error fetching OMNI timestamps: {e}")
    raise SystemExit(1)

sdo_df = pd.DataFrame({
    "omni_time": timeindex,
    "sdo_start": (timeindex - pd.Timedelta(hours=106)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sdo_end": (timeindex - pd.Timedelta(hours=86)).strftime("%Y-%m-%dT%H:%M:%SZ"),
})

sdo_path = os.path.join(DATADIR, SDO_CSV)
sdo_df.to_csv(sdo_path, index=False)

print(f"SDO query records saved: {sdo_path} ({len(sdo_df)} records)")
print("Run processing.py next - it will download & process images on-demand!")
