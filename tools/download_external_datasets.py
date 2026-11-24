"""Downloader script for Phase 4 Global Datasets
Downloads official open data:
1. UK DfT STATS19 (2022 Collisions, Casualties, Vehicles) - OGL v3.0
2. France BAAC (2023 Caractéristiques, Lieux, Usagers, Véhicules) - Open Licence 2.0
3. GB Traffic Counts (AADF) - OGL v3.0
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

DOWNLOADS = [
    # 1. UK STATS19 2022
    {
        "name": "UK STATS19 2022 Collisions",
        "url": "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2022.csv",
        "dest": EXTERNAL_DIR / "uk_stats19" / "dft-road-casualty-statistics-collision-2022.csv",
        "is_zip": False
    },
    {
        "name": "UK STATS19 2022 Casualties",
        "url": "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-casualty-2022.csv",
        "dest": EXTERNAL_DIR / "uk_stats19" / "dft-road-casualty-statistics-casualty-2022.csv",
        "is_zip": False
    },
    {
        "name": "UK STATS19 2022 Vehicles",
        "url": "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-2022.csv",
        "dest": EXTERNAL_DIR / "uk_stats19" / "dft-road-casualty-statistics-vehicle-2022.csv",
        "is_zip": False
    },
    # 2. France BAAC 2023
    {
        "name": "France BAAC 2023 Caractéristiques",
        "url": "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241028-103125/caract-2023.csv",
        "dest": EXTERNAL_DIR / "france_baac" / "caract-2023.csv",
        "is_zip": False
    },
    {
        "name": "France BAAC 2023 Lieux",
        "url": "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153219/lieux-2023.csv",
        "dest": EXTERNAL_DIR / "france_baac" / "lieux-2023.csv",
        "is_zip": False
    },
    {
        "name": "France BAAC 2023 Usagers",
        "url": "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153328/usagers-2023.csv",
        "dest": EXTERNAL_DIR / "france_baac" / "usagers-2023.csv",
        "is_zip": False
    },
    {
        "name": "France BAAC 2023 Véhicules",
        "url": "https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153253/vehicules-2023.csv",
        "dest": EXTERNAL_DIR / "france_baac" / "vehicules-2023.csv",
        "is_zip": False
    },
    # 3. GB Traffic Counts (AADF)
    {
        "name": "GB Traffic Counts AADF",
        "url": "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_aadf.zip",
        "dest": EXTERNAL_DIR / "gb_traffic" / "dft_traffic_counts_aadf.zip",
        "is_zip": True,
        "extract_to": EXTERNAL_DIR / "gb_traffic"
    }
]

def download_file(item):
    name = item["name"]
    url = item["url"]
    dest = Path(item["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[EXISTS] {name} ({dest.stat().st_size / (1024*1024):.2f} MB) -> {dest.name}")
        return

    print(f"[DOWNLOADING] {name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(dest, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                print(f"  Downloaded {total / (1024*1024):.1f} MB...", end="\r")
    print(f"\n[DONE] {name} saved to {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")

    if item.get("is_zip") and item.get("extract_to"):
        print(f"[EXTRACTING] {dest.name} to {item['extract_to']}...")
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall(item["extract_to"])
        print(f"[EXTRACTED] {dest.name}")

if __name__ == "__main__":
    for item in DOWNLOADS:
        try:
            download_file(item)
        except Exception as e:
            print(f"[ERROR] Failed downloading {item['name']}: {e}")
