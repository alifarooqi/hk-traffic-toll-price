#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hong Kong Traffic Data ETL Script
---------------------------------
- Downloads historical detector XML snapshots (resumable)
- Parses and aggregates by corridor
- Labels before/after 31 May 2025 toll policy change
- Adds peak/off-peak time buckets
- Outputs hourly traffic CSV

Usage:
    python traffic_etl.py
"""

import os
import glob
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
START_DATE = datetime(2025, 5, 1)
END_DATE   = datetime(2025, 6, 30)
POLICY_CHANGE_DATE = datetime(2025, 5, 31)

LIST_API   = "https://api.data.gov.hk/v1/historical-archive/list-file-versions"
GET_API    = "https://api.data.gov.hk/v1/historical-archive/get-file"
RAW_URL    = "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"

DATA_DIR   = "xml_cache"   # local cache folder
CACHE_FILE = "hk_tunnel_traffic.csv"

LOCATIONS_CSV_URL = "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/traffic_speed_volume_occ_info.csv"


# -------------------------
# Load detector metadata
# -------------------------
def load_metadata():
    loc_df = pd.read_csv(LOCATIONS_CSV_URL)
    loc_df.columns = [c.strip() for c in loc_df.columns]

    corridor_keywords = {
        "tai_lam": ["Tai Lam"],
        "nt_circular": ["Tuen Mun Road", "NT Circular"],
    }

    def map_corridor(road_name):
        for corridor, kws in corridor_keywords.items():
            for kw in kws:
                if kw.lower() in str(road_name).lower():
                    return corridor
        return None

    loc_df["corridor"] = loc_df["Road_EN"].apply(map_corridor)
    loc_corridors = loc_df.dropna(subset=["corridor"])
    valid_ids = set(loc_corridors["AID_ID_Number"].astype(str))

    print("✅ Corridors mapped:", loc_corridors["corridor"].unique())
    return loc_corridors, valid_ids


# -------------------------
# Download snapshots (resumable)
# -------------------------
def download_snapshots():
    os.makedirs(DATA_DIR, exist_ok=True)
    curr = START_DATE

    while curr <= END_DATE:
        day = curr.strftime("%Y%m%d")
        resp = requests.get(LIST_API, params={"url": RAW_URL, "start": day, "end": day})
        versions = resp.json().get("timestamps", [])
        files = glob.glob(os.path.join(DATA_DIR, f"{day}-*.xml")) + glob.glob(os.path.join(DATA_DIR, f"{day}-*.skip"))
        downloaded = {os.path.splitext(os.path.basename(f))[0] for f in files}
        to_download = [ver for ver in versions if ver not in downloaded]
        print(f"{curr.date()} -> {len(versions)} total snapshots, {len(to_download)} missing, {len(downloaded)} already cached.")
        for ts in tqdm(to_download, desc=f"Downloading {day}"):
            fname = os.path.join(DATA_DIR, f"{ts}.xml")
            if os.path.exists(fname):
                continue
            r = requests.get(GET_API, params={"url": RAW_URL, "time": ts})
            if r.status_code == 200:
                with open(fname, "wb") as f:
                    f.write(r.content)
            else:
                with open(fname.replace(".xml", ".skip"), "w") as f:
                    f.write("not found")

        curr += timedelta(days=1)
    print("✅ All snapshots downloaded to", DATA_DIR)


# -------------------------
# Parse one XML snapshot
# -------------------------
def parse_snapshot_file(filepath, valid_ids):
    out = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        date_text = root.findtext(".//date")
        for period in root.findall(".//period"):
            ts_str = f"{date_text} {period.findtext('period_from')}"
            for det in period.findall(".//detector"):
                did = det.findtext("detector_id")
                if did not in valid_ids:
                    continue

                speed, volume, occupancy = None, None, None
                if det.find("speed") is not None:
                    speed = float(det.findtext("speed") or 0)
                    volume = int(det.findtext("volume") or 0)
                    occupancy = float(det.findtext("occupancy") or 0)
                else:
                    lanes = det.findall(".//lane")
                    lane_speeds, lane_occ, lane_vols = [], [], []
                    for ln in lanes:
                        if ln.find("speed") is not None:
                            lane_speeds.append(float(ln.findtext("speed") or 0))
                        if ln.find("occupancy") is not None:
                            lane_occ.append(float(ln.findtext("occupancy") or 0))
                        if ln.find("volume") is not None:
                            lane_vols.append(int(ln.findtext("volume") or 0))
                    if lane_speeds:
                        speed = sum(lane_speeds) / len(lane_speeds)
                    if lane_occ:
                        occupancy = sum(lane_occ) / len(lane_occ)
                    if lane_vols:
                        volume = sum(lane_vols)

                out.append({
                    "timestamp": ts_str,
                    "detector_id": did,
                    "speed": speed,
                    "volume": volume,
                    "occupancy": occupancy
                })
    except Exception:
        return []
    return out


# -------------------------
# Parse all cached XMLs
# -------------------------
def parse_all_snapshots(valid_ids):
    rows = []
    files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
    print(f"📖 Parsing {len(files)} XML snapshots...")
    for filepath in tqdm(files, desc="Parsing XMLs"):
        rows.extend(parse_snapshot_file(filepath, valid_ids))
    print("✅ Parsed rows:", len(rows))
    return rows


# -------------------------
# Aggregate & add buckets
# -------------------------
def aggregate(rows, loc_corridors):
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.merge(
        loc_corridors[["AID_ID_Number","corridor"]],
        left_on="detector_id", right_on="AID_ID_Number", how="left"
    )

    df_hour = df.groupby([pd.Grouper(key="timestamp", freq="H"), "corridor"]).agg(
        volume=("volume","sum"),
        speed=("speed","mean"),
        occupancy=("occupancy","mean")
    ).reset_index()

    pivot = df_hour.pivot(index="timestamp", columns="corridor", values="volume").fillna(0).reset_index()

    # Add before/after policy bucket
    pivot["period"] = pivot["timestamp"].apply(lambda x: "before" if x < POLICY_CHANGE_DATE else "after")

    # Add peak/off-peak bucket
    def is_peak(ts):
        hm = ts.hour*100 + ts.minute
        return ((715 <= hm <= 945) or (1715 <= hm <= 1900))

    pivot["slot"] = pivot["timestamp"].apply(
        lambda x: "peak" if is_peak(x) else "offpeak" if x >= POLICY_CHANGE_DATE else "na"
    )

    pivot.to_csv(CACHE_FILE, index=False)
    print("✅ Aggregated data saved:", CACHE_FILE)
    return pivot


# -------------------------
# Main
# -------------------------
def main():
    loc_corridors, valid_ids = load_metadata()
    download_snapshots()
    rows = parse_all_snapshots(valid_ids)
    pivot = aggregate(rows, loc_corridors)
    print(pivot.head())


if __name__ == "__main__":
    main()
