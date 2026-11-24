"""International Dataset Acquisition Script for Global Accident Intelligence.
Downloads and validates official open datasets for:
1. Germany: Unfallatlas 2022 (Statistikportal / OpenGeodata NRW) - dl-de/by-2-0
2. Australia: Victoria Road Crash Data Flat File (Transport Victoria) - CC BY 4.0
3. Canada: National Collision Database 2019 (Transport Canada) - OGL-Canada
"""
import os
import sys
import time
import zipfile
import io
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def download_file(url: str, dest_path: Path, is_zip: bool = False, extract_to: Path = None):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"  [ALREADY EXISTS] {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return dest_path

    print(f"  Downloading from {url}...")
    t0 = time.time()
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read()
    
    elapsed = time.time() - t0
    print(f"  Downloaded {len(content) / (1024*1024):.2f} MB in {elapsed:.2f}s ({(len(content)/1024/1024)/max(elapsed, 0.1):.2f} MB/s)")

    if is_zip and extract_to:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(extract_to)
            print(f"  Extracted {len(z.namelist())} files to {extract_to}")
            for name in z.namelist():
                print(f"    - {name}")
        # Save archive as well
        with open(dest_path, "wb") as f:
            f.write(content)
    else:
        with open(dest_path, "wb") as f:
            f.write(content)
        print(f"  Saved to {dest_path}")
    
    return dest_path

print("=" * 80)
print("ACQUIRING OFFICIAL INTERNATIONAL ROAD SAFETY DATASETS")
print("=" * 80)

# 1. GERMANY: Unfallatlas 2022
print("\n[1] GERMANY: Destatis Unfallatlas 2022 (dl-de/by-2-0)...")
de_zip_url = "https://www.opengeodata.nrw.de/produkte/transport_verkehr/unfallatlas/Unfallorte2022_EPSG25832_CSV.zip"
de_dir = EXTERNAL_DIR / "germany_unfallatlas"
download_file(de_zip_url, de_dir / "Unfallorte2022_CSV.zip", is_zip=True, extract_to=de_dir)

# 2. AUSTRALIA: Victoria Road Crash Data
print("\n[2] AUSTRALIA: Transport Victoria Road Crash Data (CC BY 4.0)...")
aus_url = "https://opendata.transport.vic.gov.au/dataset/bb77800e-1857-4edc-bf9e-e188437a1c8e/resource/5df1f373-0c90-48f5-80e1-7b2a35507134/download/victorian_road_crash_data.csv"
aus_dir = EXTERNAL_DIR / "australia_victoria"
download_file(aus_url, aus_dir / "victorian_road_crash_data.csv")

# 3. CANADA: National Collision Database 2019
print("\n[3] CANADA: Transport Canada NCDB 2019 (OGL-Canada)...")
can_url = "https://open.canada.ca/data/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a/resource/1e2a6aa0-c35f-4736-984a-71f2eac1c1dc/download/2019_dataset_en.csv"
can_dir = EXTERNAL_DIR / "canada_ncdb"
download_file(can_url, can_dir / "ncdb_2019.csv")

print("\n" + "=" * 80)
print("DOWNLOADS COMPLETE")
print("=" * 80)
