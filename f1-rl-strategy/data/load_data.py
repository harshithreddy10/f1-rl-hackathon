import fastf1
import pandas as pd
import numpy as np
import os
import pickle

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def load_monaco_2024():
    """Load Monaco 2024 race session."""
    print("Loading Monaco 2024 Race... (first run takes ~5 mins)")
    session = fastf1.get_session(2024, "Monaco", "R")
    session.load()
    print("session loaded !")
    return session

def extract_driver_laps(session,driver="LEC"):
    laps = session.laps.pick_driver(driver).copy()
    df = pd.DataFrame()
    df["LapNumber"]   = laps["LapNumber"].values
    df["Compound"]    = laps["Compound"].values
    df["TyreLife"]    = laps["TyreLife"].values
    df["LapTime_s"]   = laps["LapTime"].dt.total_seconds().values
    df["PitIn"]       = laps["PitInTime"].notna().astype(int).values
    df["PitOut"]      = laps["PitOutTime"].notna().astype(int).values
    df["TrackStatus"] = laps["TrackStatus"].values
    
    df = df.dropna(subset=["LapTime_s"]).reset_index(drop=True)
    return df

def build_dataset(session,drivers= None):
    if drivers is None :
        drivers = [session.results.iloc[i]["Abbreviation"] for i in range(10)]
    all_laps = []
    
    for drv in drivers : 
        try :
            df = extract_driver_laps(session, drv)
            df["Driver"] = drv
            all_laps.append(df)
            print(f"  ✅ {drv}: {len(df)} laps")
        except Exception as e:
            print(f"  ⚠️  {drv}: skipped ({e})")
        
    dataset = pd.concat(all_laps, ignore_index=True)
    return dataset

def save_dataset(df, path="data/monaco_2024_laps.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Dataset saved → {path}  ({len(df)} rows)")


if __name__ == "__main__":
    session = load_monaco_2024()
    df = build_dataset(session)
    save_dataset(df)
    print("\n📊 Sample:")
    print(df.head(10).to_string())