import os
import numpy as np
import pandas as pd
import glob
from sunpy.map import Map


YEAR = 2017
DATADIR = "data"
OUTDIR = "C"  


os.makedirs(OUTDIR, exist_ok=True)


OMNICSV = os.path.join(DATADIR, f"omni2_{YEAR}.csv")
ICMECSV = os.path.join(DATADIR, f"icmecat_{YEAR}.csv")


def load_omni():
    df = pd.read_csv(OMNICSV, header=None)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.iloc[:, 0])
    ncols = df.shape[1]
    df.columns = ["time"] + [f"col{i}" for i in range(1, ncols)]
    df = df.drop(columns=["time"])

    speed = df["col1"].to_numpy()      
    density = df["col5"].to_numpy()   
    temp = df["col6"].to_numpy()      
    pressure = df["col8"].to_numpy()  
    sigmaB = df["col15"].to_numpy()   

    return df.index, speed, density, temp, pressure, sigmaB


def load_icme_flags(timeindex):
    ic = pd.read_csv(ICMECSV)
    ic = ic[ic["sc_insitu"].str.contains("Wind", case=False, na=False)].copy()

    ic["starttime"] = pd.to_datetime(ic["icme_starttime"])
    ic["endtime"] = pd.to_datetime(ic["moendtime"])

    yearstart = pd.Timestamp(f"{YEAR}-01-01T00:00:00Z")
    yearend = pd.Timestamp(f"{YEAR}-12-31T23:59:59Z")

    ic = ic[(ic["endtime"] > yearstart) & (ic["starttime"] < yearend)]

    flags = np.zeros(len(timeindex), dtype=float)

    for _, row in ic.iterrows():
        s = max(row["starttime"], yearstart)
        e = min(row["endtime"], yearend)
        flags[(timeindex >= s) & (timeindex <= e)] = 1.0

    return flags


def load_coronal_hole(timeindex):
    AIA_ROOT = os.path.join(DATADIR, "aia_193", str(YEAR), f"{YEAR}_full")  
    
    t_img_min = timeindex.min() - pd.Timedelta(hours=106)
    t_img_max = timeindex.max() - pd.Timedelta(hours=86)
    img_times = pd.date_range(t_img_min.floor('H'), t_img_max.floor('H'), freq='H')
    
    print("Loading all AIA images...")
    img_sums = {}
    for t_img in img_times:
        pattern = os.path.join(AIA_ROOT, f"*AIA*193*{t_img.strftime('%Y%m%d_%H%M')}*fits")  
        files = glob.glob(pattern)
        if files:
            try:
                m = Map(files[0])
                data = m.data.astype(np.float32)  
                img_sums[t_img] = np.sum(data < 15)
            except Exception as e:
                print(f"Skip {t_img}: {e}")
    
    areas = np.zeros(len(timeindex), dtype=np.float32)
    for i, t in enumerate(timeindex):
        window_start = t - pd.Timedelta(hours=106)
        window_times = pd.date_range(window_start, window_start + pd.Timedelta(hours=20), freq='H')
        window_sums = [img_sums.get(tt, np.nan) for tt in window_times]
        areas[i] = np.nanmean(window_sums)  
    
    lag = 96
    coronal = np.roll(areas, lag)
    coronal[:lag] = areas[:lag]  
    
    return coronal


def save_series(path, arr):
    np.savetxt(path, arr, fmt="%.6f")  


def main():
    timeindex, speed, density, temp, pressure, sigmaB = load_omni()
    icme_flags = load_icme_flags(timeindex)
    area = load_coronal_hole(timeindex)

    files = [
        ("speedtrain.txt", speed),
        ("areatrain15.txt", area),
        ("densitytrain.txt", density),
        ("temptrain.txt", temp),
        ("pressuretrain.txt", pressure),
        ("sigmaBtrain.txt", sigmaB),
        ("ICMEtrain.txt", icme_flags),
    ]

    for fname, data in files:
        save_series(os.path.join(OUTDIR, fname), data)
        print(f"Saved {fname}: {len(data)} values")


if __name__ == "__main__":
    main()
