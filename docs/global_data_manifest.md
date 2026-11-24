# Global Data Source & Licence Manifest

**Platform:** Accident Intelligence SaaS  
**Phase:** Phase 4 - Global Data Expansion  
**Date:** November 2026  
**Status:** Ingested & Verified  

---

## 1. Executive Summary & Commercial Readiness

Phase 4 establishes an open, legally unencumbered data foundation for international accident intelligence. Unlike the Phase 2 US Accidents training dataset (restricted by CC BY-NC-SA 4.0 non-commercial clauses), **all Phase 4 data sources carry permissive open government licences and open data licences that explicitly authorize commercial exploitation, adaptation, and SaaS deployment**.

---

## 2. Ingested Data Sources & Provenance

### 2.1 UK STATS19 Road Safety Data
- **Originating Body:** Department for Transport (DfT), United Kingdom
- **Dataset Title:** Road Accident and Safety Statistics (STATS19)
- **Files Ingested:**
  - `dft-road-casualty-statistics-collision-2022.csv` (19.18 MB, 106,004 records)
  - `dft-road-casualty-statistics-casualty-2022.csv` (10.72 MB, 135,480 records)
  - `dft-road-casualty-statistics-vehicle-2022.csv` (20.34 MB)
- **Geographic Footprint:** Great Britain (England, Scotland, Wales)
- **Temporal Coverage:** Full Calendar Year 2022
- **Licence:** [Open Government Licence v3.0 (OGL v3.0)](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
- **Commercial Use:** **Permitted** (World-wide, royalty-free, perpetual license to use, adapt, and exploit commercially).
- **Mandatory Attribution Statement:**
  > *"Contains Department for Transport data licensed under the Open Government Licence v3.0."*

---

### 2.2 France BAAC Road Safety Data
- **Originating Body:** Ministère de l'Intérieur / Observatoire National Interministériel de la Sécurité Routière (ONISR), France
- **Dataset Title:** Bases de données annuelles des accidents corporels de la circulation routière (BAAC)
- **Files Ingested:**
  - `caract-2023.csv` (6.29 MB, 54,823 accidents)
  - `lieux-2023.csv` (6.84 MB, 70,861 roadway records)
  - `usagers-2023.csv` (13.29 MB, 125,790 casualty records)
  - `vehicules-2023.csv` (6.13 MB, 94,435 vehicle records)
- **Geographic Footprint:** France (Metropolitan France + Overseas Departments/Territories)
- **Temporal Coverage:** Full Calendar Year 2023
- **Licence:** [Licence Ouverte / Open Licence v2.0 (Etalab)](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
- **Commercial Use:** **Permitted** (Free, open, worldwide license for commercial and non-commercial reuse, transformation, and distribution).
- **Mandatory Attribution Statement:**
  > *"Ministère de l'Intérieur / ONISR - Données des accidents corporels de la circulation routière - Licence Ouverte v2.0."*

---

### 2.3 GB Traffic Counts (AADF - Annual Average Daily Flow)
- **Originating Body:** Department for Transport (DfT), United Kingdom
- **Dataset Title:** Street-Level Traffic Counts (Average Annual Daily Flow)
- **Files Ingested:**
  - `dft_traffic_counts_aadf.csv` (142.96 MB, 600,551 link-year observations; 22,240 active count points in 2022)
- **Geographic Footprint:** Great Britain road network (Major and Minor road networks)
- **Temporal Coverage:** 2000–2023 (Active benchmark year: 2022)
- **Licence:** [Open Government Licence v3.0 (OGL v3.0)](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
- **Commercial Use:** **Permitted**.
- **Mandatory Attribution Statement:**
  > *"Contains Department for Transport traffic count data licensed under the Open Government Licence v3.0."*

---

### 2.4 Copernicus ERA5 Reanalysis Historical Weather
- **Originating Body:** European Centre for Medium-Range Weather Forecasts (ECMWF) on behalf of the European Union
- **Dataset Title:** ERA5 Reanalysis Single Levels Hourly Timeseries
- **Access Protocol:** Open-access Copernicus API / Open-Meteo ERA5 Reanalysis Archive
- **Variables:** 2m air temperature (°C), relative humidity (%), surface pressure (hPa), 10m wind speed (km/h), precipitation rate (mm).
- **Geographic Footprint:** Global ($0.25^\circ \times 0.25^\circ$ grid, WGS84)
- **Temporal Coverage:** 1940–Present
- **Licence:** [Copernicus Climate Change Service Open Access / CC-BY 4.0](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries)
- **Commercial Use:** **Permitted**.
- **Mandatory Attribution Statement:**
  > *"Generated using Copernicus Climate Change Service information [2026]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."*

---

### 2.5 OpenStreetMap Road Network Taxonomy
- **Originating Body:** OpenStreetMap Foundation (OSMF)
- **Schema File:** `data/external/osm_road_network/osm_highway_tag_mapping.json`
- **Application:** Universal highway classification mapping (`motorway`, `trunk`, `primary`, `secondary`, `residential`) and roadway infrastructure tags (`highway=traffic_signals`, `junction=roundabout`, `highway=crossing`).
- **Geographic Footprint:** Global
- **Licence:** [Open Data Commons Open Database License (ODbL) 1.0](https://www.openstreetmap.org/copyright)
- **Commercial Use:** **Permitted** (Share-Alike applies to derivative database distributions; internal model training is permitted).
- **Mandatory Attribution Statement:**
  > *"© OpenStreetMap contributors."*

---

## 3. Licence Comparison & SaaS Commercial Clearance Matrix

| Data Source | Jurisdiction | Primary Licence | Commercial SaaS Use Allowed? | Can Charge End Users? | Attribution Required? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UK STATS19** | United Kingdom | OGL v3.0 | **YES** | **YES** | YES |
| **France BAAC** | France | Licence Ouverte v2.0 | **YES** | **YES** | YES |
| **GB Traffic AADF** | United Kingdom | OGL v3.0 | **YES** | **YES** | YES |
| **Copernicus ERA5** | European Union / Global | CC-BY 4.0 | **YES** | **YES** | YES |
| **OpenStreetMap** | Global | ODbL 1.0 | **YES** | **YES** | YES |
| *US Accidents (Moosavi)* | United States | *CC BY-NC-SA 4.0* | **NO (Research Only)**| **NO** | YES |

> [!IMPORTANT]
> The addition of UK STATS19 and France BAAC provides the first fully commercial-cleared training and validation corpora for the platform, establishing a path to commercialize risk and human injury severity without breaching third-party IP rights.
