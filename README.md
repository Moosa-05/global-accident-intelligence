<div align="center">

# 🌐 Global Accident Intelligence

**An open-source, full-stack machine learning & spatial intelligence platform for cross-national road traffic safety, collision injury severity estimation, and infrastructure risk analytics.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%20%7C%2019-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-v4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-FF7043)](https://lightgbm.readthedocs.io)
[![Backend Tests](https://img.shields.io/badge/pytest-64%20passed-success?logo=pytest&logoColor=white)](https://docs.pytest.org)
[![Frontend Tests](https://img.shields.io/badge/vitest-8%20passed-success?logo=vitest&logoColor=white)](https://vitest.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-OGL%20%2F%20LO%20%2F%20CC%20BY--NC--SA-lightgrey)](#-data-provenance--licensing-matrix)

[Explore Documentation](#-system-architecture) • [Models & Benchmarks](#-predictive-intelligence-pipelines) • [API Specification](#-rest-api-endpoints) • [Local Setup](#-local-setup--quickstart)

</div>

---

## 📋 Table of Contents

- [Overview & Problem Statement](#-overview--problem-statement)
- [System Architecture](#-system-architecture)
- [Predictive Intelligence Pipelines](#-predictive-intelligence-pipelines)
  - [1. Collision-Level Human Injury Severity](#1-collision-level-human-injury-severity-multi-country)
  - [2. Macro-Scale Annual Road Segment Exposure Risk](#2-macro-scale-annual-road-segment-exposure-risk)
  - [3. Real-Time Freeway Corridor Sensor Risk](#3-real-time-freeway-corridor-sensor-risk)
  - [4. Traffic Disruption Delay Severity](#4-traffic-disruption-delay-severity-research-benchmark)
- [Scientific Methodology & Empirical Validation](#-scientific-methodology--empirical-validation)
  - [Leave-One-Country-Out (LOCO) Generalization](#leave-one-country-out-loco-generalization)
  - [Coordinate-Free Invariance](#coordinate-free-invariance)
  - [Calibrated Trauma Triage (Decision Modes)](#calibrated-trauma-triage-decision-modes)
- [Data Provenance & Licensing Matrix](#-data-provenance--licensing-matrix)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [REST API Endpoints](#-rest-api-endpoints)
- [Local Setup & Quickstart](#-local-setup--quickstart)
  - [Prerequisites](#prerequisites)
  - [Manual Execution](#manual-execution)
  - [Docker Compose Deployment](#docker-compose-deployment)
  - [Running Test Suites](#running-test-suites)
- [System Limitations & Ethical Disclaimers](#-system-limitations--ethical-disclaimers)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 🌟 Overview & Problem Statement

Traffic collisions claim over **1.19 million lives annually worldwide** (WHO Global Status Report on Road Safety). Traditional statistical safety analysis relies on retrospective, single-jurisdiction historical records that often fail to generalize across borders due to:
1. **Disparate Administrative Classifications:** Different thresholds for inpatient hospitalization between jurisdictions (e.g., France/Germany $\ge 24\text{ hours}$ vs. UK/Australia broader inpatient definitions).
2. **Spatial Overfitting & Memorization:** Spatial tree-based models relying on raw GPS coordinates inadvertently memorize local bounding boxes rather than underlying physical accident dynamics.
3. **Temporal Scale Mismatch:** Lack of distinction between immediate CAD dispatch triage, real-time corridor shockwave monitoring, and long-term capital infrastructure planning.

**Global Accident Intelligence** resolves these challenges by introducing a **multi-scale, multi-country AI engine** grounded in open government accident censuses (United Kingdom, France, Germany, Australia, Canada, United States). It decouples spatial geography from physical risk, enforces dual Imperial/Metric unit invariance, and exposes specialized inference services for dispatch triage, corridor telematics, and infrastructure investment.

---

## 🏛 System Architecture

The platform follows a decoupled, production-grade microservices pattern with cryptographic session isolation, ensuring seamless anonymous demonstration access alongside strict multi-tenant SaaS capabilities:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CLIENT TIER (React 18/19 + Vite)                     │
│  - Dark Graphite UI (#0f172a) with Amber & Muted Emerald Accents      │
│  - MapLibre GL / Leaflet Interactive Spatial GIS Intelligence          │
│  - Imperial / Metric Real-Time Unit Normalization Engine               │
│  - Recharts Dynamic Calibration Curves & Telematics Visualizers        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / WebSocket (:3000 -> :80)
┌───────────────────────────────────▼────────────────────────────────────┐
│              EDGE PROXY & REVERSE PROXY (Nginx High-Perf)              │
│  - TLS Termination, Gzip/Brotli Compression, Security Headers (CSP)    │
│  - Static Asset Caching & SPA History Mode Route Dispatching           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Reverse Proxy (:8000)
┌───────────────────────────────────▼────────────────────────────────────┐
│              BACKEND API CORE (FastAPI + Pydantic v2)                  │
│  - Cryptographic Session Isolation for Anonymous Live Demo Users       │
│  - IP-based Sliding-Window Rate Limiting (SlowAPI / In-Memory Token)   │
│  - Physics-Aware Unit Conversion (Speed, Temp, Pressure, Distance)     │
│  - Commercial Licensing Gatekeeper (CC BY-NC-SA 4.0 Enforcement)      │
└───────────────┬───────────────────┬───────────────────┬────────────────┘
                │                   │                   │
    ┌───────────▼─────────┐ ┌───────▼─────────┐ ┌───────▼────────────┐
    │  PERSISTENCE TIER   │ │ INFERENCE TIER  │ │ OBSERVABILITY TIER │
    │ - PostgreSQL 15     │ │ - LightGBM GBDT │ │ - MLflow Registry  │
    │ - SQLite Fallback   │ │ - Poisson/NB GLM│ │ - Alembic Migrator │
    │ - SQLAlchemy 2.0    │ │ - Scikit-Learn  │ │ - Health Probes    │
    └─────────────────────┘ └─────────────────┘ └────────────────────┘
```

---

## 🔬 Predictive Intelligence Pipelines

The platform maintains four production machine learning pipelines across different spatial and temporal scales:

### 1. Collision-Level Human Injury Severity (Multi-Country)
- **Task:** 3-Class physiological trauma severity classification (`SLIGHT`, `SERIOUS`, `FATAL`).
- **Operational Horizon:** CAD dispatch triage and emergency responder resource allocation (pre-arrival).
- **Deployed Production Model (V1):** LightGBM classifier trained on UK DfT STATS19 and French ONISR BAAC corpora (UK In-Country Macro F1 = `0.4065`, France In-Country Macro F1 = `0.3934`).
- **Research Candidate Model (V2):** 4-Country Coordinate-Free LightGBM trained on `618,571` harmonized records across the UK, France, Germany, and Australia (Global Macro F1 = `0.4508`, Temperature-Scaled Expected Calibration Error $\text{ECE} = 0.0291$).
- **Dual Operational Decision Modes:**
  - **Mode A (Statistical Calibration):** Temperature-scaled argmax ($T = 0.6122$) optimizing overall Macro F1 and probabilistic truthfulness across general accident queries.
  - **Mode B (Trauma Life-Safety Screening):** Calibrated non-argmax decision thresholds ($\tau_{\text{fatal}} = 0.20, \tau_{\text{serious}} = 0.35$), elevating **Fatal Recall to 59.62%** and Serious Recall to 57.77% on untouched test holdouts to prevent under-triage of catastrophic collisions.

### 2. Macro-Scale Annual Road Segment Exposure Risk
- **Task:** Predicts expected annual collision frequency $\mathbb{E}[Y]$ and crash rate per Million Vehicle-Kilometres Travelled (MVKT).
- **Operational Horizon:** Long-term civil infrastructure planning, Vision Zero road safety audits, and municipal capital budget allocation.
- **Model:** Negative Binomial / Poisson Generalized Linear Models (GLM) and LightGBM with explicit exposure offset formulation:
  $$\log(\mu_i) = \beta_0 + \sum_j \beta_j X_{ij} + \log\left(\frac{\text{Length}_i \times \text{AADF}_i \times 365}{1\,000\,000}\right)$$
- **Data Foundation:** 19,414 UK DfT count points (AADF census) and 2,429 French National Road network sections (TMJA census).

### 3. Real-Time Freeway Corridor Sensor Risk
- **Task:** Binary classification of crash likelihood in the subsequent 15-minute time window along a highway link.
- **Operational Horizon:** Active Traffic Management (variable speed limit sign adjustments, ramp metering, emergency roadside dispatch).
- **Model:** LightGBM calibrated on 18,059 15-minute aggregate traffic sensor intervals from the Tennessee Department of Transportation (TDOT) Chattanooga sensor grid (I-24, I-75, SR-153).
- **Key Predictors:** Moving average speed, traffic volume, lane occupancy, sudden deceleration shockwaves (`speed_drop`), and spatial speed differentials across adjacent sensor stations.

### 4. Traffic Disruption Delay Severity (Research Benchmark)
- **Task:** 4-Class post-incident traffic delay severity estimation (Classes 1 through 4) conditional on a collision occurring.
- **Operational Horizon:** Post-incident traffic routing, navigation delay modeling, and incident clearing dispatch.
- **Model:** LightGBM Multiclass with spatial and weather features.
- **Licensing Constraint:** Strictly restricted to non-commercial academic research use under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

---

## 📊 Scientific Methodology & Empirical Validation

### Leave-One-Country-Out (LOCO) Generalization

To rigorously assess geographic transferability without data leakage, models were evaluated on entire sovereign nations completely held out from feature engineering, hyperparameter tuning, and model fitting:

| Evaluation Target Country | Sample Size ($n$) | Training Split Countries | V1 Baseline Macro F1 | V2 Candidate Macro F1 | Improvement ($\Delta$) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Australia (Victoria)** | 201,275 | Germany (DE), France (FR), UK (GB) | 0.1894 | **0.3690** | **+0.1796** |
| **Germany (Destatis)** | 256,492 | Australia (AU), France (FR), UK (GB) | 0.2353 | **0.3095** | **+0.0742** |
| **France (ONISR BAAC)** | 54,822 | Australia (AU), Germany (DE), UK (GB) | 0.3934* | **0.3230** | Domain Gap Withheld |
| **United Kingdom (STATS19)** | 105,982 | Australia (AU), Germany (DE), France (FR) | 0.4065* | **0.1714** | Domain Gap Withheld |

*\*Note: V1 Baseline for France and UK represents in-country training. When UK or France are withheld entirely from training, severe domain transfer penalties occur, confirming why V1 remains the deployed production model for UK/France while V2 serves as the multi-country cross-national benchmark.*

### Coordinate-Free Invariance

Traditional geospatial machine learning models often memorize geographic coordinates (e.g., latitude and longitude splits in decision trees). In Global Accident Intelligence:
- **Global models strictly omit raw latitude and longitude** from the feature vector.
- Tree splits are forced to learn from physical invariants: road hierarchy, speed regime, vehicle mass ratio (commercial vs. passenger), lighting conditions, junction conflict geometry, and meteorological reanalysis parameters.
- This prevents artificial extreme-hazard predictions when querying coordinates in the Southern Hemisphere or across non-training territories.

### Calibrated Trauma Triage (Decision Modes)

In emergency medical dispatch, a false negative (classifying a fatal crash as slight) carries catastrophic societal consequences compared to a false positive. We implemented validation-optimized threshold tuning on an untouched 10% test split ($n = 61,858$):

```
Standard Argmax (Mode A):
  Fatal Recall:   8.41%   │ Fatal Precision: 41.38% │ Macro F1: 0.4508
  Serious Recall: 32.14%  │ Overall Accuracy: 74.8%

Life-Safety Triage (Mode B: τ_fatal = 0.20, τ_serious = 0.35):
  Fatal Recall:   59.62%  │ Fatal Precision: 6.26%  │ Macro F1: 0.3534
  Serious Recall: 57.77%  │ Serious Precision: 38.1%
```

---

## 📜 Data Provenance & Licensing Matrix

All datasets used in Global Accident Intelligence originate from official government portals or peer-reviewed academic repositories:

| Dataset | Jurisdictional Authority / Source | Records | Commercial Clearance | License | Provenance / Citation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **STATS19 Road Safety Data** | United Kingdom Department for Transport (DfT) | 105,982 | **Yes** | Open Government Licence v3.0 (OGL) | Contains public sector information licensed under the Open Government Licence v3.0. |
| **BAAC Accident Database** | France Ministère de l'Intérieur / ONISR | 54,822 | **Yes** | Licence Ouverte v2.0 (Etalab) | Données publiques ouvertes du Ministère de l'Intérieur français. |
| **Unfallatlas 2022** | Germany Destatis / Statistische Ämter des Bundes | 256,492 | **Yes** | dl-de/by-2-0 | Datenlizenz Deutschland – Namensnennung – Version 2.0. |
| **Victorian Road Crash Data** | Australia Transport Victoria / Dept. of Transport | 202,554 | **Yes** | CC BY 4.0 | Creative Commons Attribution 4.0 International. |
| **National Collision Database** | Transport Canada | 272,301 | **Yes** | OGL - Canada | Maintained as an independent Canadian benchmark. |
| **UK AADF Traffic Flow** | United Kingdom Department for Transport (DfT) | 19,414 | **Yes** | Open Government Licence v3.0 (OGL) | National road traffic census annual average daily flow. |
| **TMJA Road Network Flow** | France DGITM | 2,429 | **Yes** | Licence Ouverte v2.0 (Etalab) | Trafic Moyen Journalier Annuel sur le réseau routier national non concédé. |
| **US Accidents Dataset** | Moosavi et al. / Academic Benchmark | 7.7M | **No** | CC BY-NC-SA 4.0 | Non-commercial academic research model only. Gated in commercial mode. |
| **ERA5 Atmospheric Reanalysis** | ECMWF / Copernicus Climate Change Service | Grids | **Yes** | Copernicus Open Data | Surface meteorological parameters (hourly grids). |

---

## 🛠 Tech Stack

### Backend Core & Machine Learning
- **Python 3.10+** — Runtime environment
- **FastAPI** — High-performance asynchronous API framework
- **Pydantic v2** — Data validation and schema enforcement
- **LightGBM** — Fast, distributed gradient boosted decision trees
- **Scikit-Learn & Statsmodels** — Preprocessing pipelines and Generalized Linear Models (Poisson / Negative Binomial)
- **SQLAlchemy 2.0 & Alembic** — ORM and schema migration engine
- **PostgreSQL 15 / SQLite** — Relational database with automatic persistent SQLite fallback
- **MLflow** — Model lineage, metrics recording, and parameter tracking

### Frontend & Data Visualization
- **React 18 / 19 & TypeScript** — Component-driven user interface
- **Vite 6** — Next-generation frontend tooling and rapid HMR
- **Tailwind CSS v4** — Modern utility-first styling with dark theme support
- **MapLibre GL & Leaflet.js** — Interactive WebGL-accelerated mapping
- **Recharts** — Responsive charting for probability distributions and sensor trends
- **Lucide React** — Consistent icon taxonomy

### DevOps & Infrastructure
- **Docker & Docker Compose** — Containerized multi-service deployment
- **Nginx** — Reverse proxy, caching, and static web hosting
- **Pytest** — Automated backend test suite (64 unit, integration, and E2E tests)
- **Vitest & React Testing Library** — Automated frontend UI test suite

---

## 📁 Repository Structure

```
global-accident-intelligence/
├── backend/
│   ├── alembic/                # Database migration versions (001-005)
│   ├── app/
│   │   ├── adapters/           # Multi-country data ingestion adapters
│   │   ├── api/v1/endpoints/   # Modular API routes (injury, exposure, corridor, auth)
│   │   ├── core/               # Security, units, licensing, rate-limiting
│   │   ├── db/models/          # SQLAlchemy ORM schemas
│   │   ├── schemas/            # Pydantic v2 request & response schemas
│   │   └── services/           # Model loading, inference, and calibration logic
│   ├── ml/                     # Model training, harmonization, and calibration scripts
│   ├── tests/                  # 64 automated pytest test cases
│   ├── Dockerfile              # Backend container specification
│   └── requirements.txt        # Python backend dependencies
├── data/
│   └── README.md               # Dataset provenance, catalog, and directory documentation
├── deployment/
│   └── nginx/                  # Production Nginx reverse proxy configurations
├── docs/                       # Technical deployment runbooks and data quality audits
├── frontend/
│   ├── src/
│   │   ├── components/         # Reusable UI controls, modals, and sidebar
│   │   ├── pages/              # Prediction views, GIS map, model registry
│   │   ├── services/           # Axios API client with automatic session headers
│   │   └── tests/              # Frontend Vitest test specifications
│   ├── Dockerfile              # Multi-stage Nginx frontend container
│   ├── package.json            # Node.js dependencies
│   └── vite.config.ts          # Vite build and proxy configuration
├── models/                     # Pre-compiled LightGBM, GLM, and preprocessor artifacts
├── docker-compose.yml          # Full-stack multi-container composition
├── requirements.txt            # Top-level Python dependencies
└── README.md                   # Platform documentation
```

---

## 🔌 REST API Endpoints

The API is fully documented via interactive OpenAPI (Swagger) specifications available at `/docs` when running the backend.

### Core Prediction Endpoints

| Method | Endpoint | Pipeline | Primary Inputs |
| :---: | :--- | :--- | :--- |
| `POST` | `/api/v1/injury/predict` | Human Injury Severity | Speed limit, road type, vehicle types, casualty count, weather, lighting |
| `POST` | `/api/v1/exposure/predict` | Annual Road Exposure | Road classification, segment length (km/mi), AADF daily traffic flow |
| `POST` | `/api/v1/predictions/predict`| Corridor Telematics Risk | Highway, milepost, speed, speed drop, traffic volume, occupancy |
| `POST` | `/api/v1/severity/predict` | Traffic Disruption | Visibility, precipitation, humidity, pressure, wind speed, junction |
| `GET`  | `/api/v1/models/` | Model Registry | Query registered models, versions, and commercial licensing clearance |

### Example Request & Response: Human Injury Severity

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/injury/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "speed_limit_mph": 60,
       "road_type": "Dual Carriageway",
       "vehicle_type": "Car",
       "weather": "Raining",
       "surface": "Wet",
       "lighting": "Darkness",
       "number_of_vehicles": 2,
       "number_of_casualties": 1,
       "decision_mode": "mode_b"
     }'
```

```json
{
  "predicted_severity": "SERIOUS",
  "probabilities": {
    "SLIGHT": 0.3842,
    "SERIOUS": 0.4418,
    "FATAL": 0.1740
  },
  "confidence": 0.4418,
  "high_fatality_risk": true,
  "model_version": "v1.0.0-stats19-baac",
  "decision_mode": "mode_b",
  "disclaimer": "Emergency dispatch life-safety screening mode enabled. Calibrated for high trauma sensitivity."
}
```

---

## 🚀 Local Setup & Quickstart

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** and **npm**
- **Git**
- *Optional:* Docker & Docker Compose

---

### Manual Execution

#### 1. Clone the Repository
```bash
git clone https://github.com/Moosa-05/global-accident-intelligence.git
cd global-accident-intelligence
```

#### 2. Backend Setup
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI backend (runs on http://127.0.0.1:8000)
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### 3. Frontend Setup
```bash
# Open a new terminal window
cd frontend

# Install Node.js dependencies
npm install

# Start Vite development server (runs on http://localhost:5173 or :3000)
npm run dev
```

---

### Docker Compose Deployment

Run the complete production stack (PostgreSQL 15, FastAPI, and Nginx-backed Frontend) with a single command:

```bash
# Build and launch all containers
docker-compose up --build -d

# Check service health
docker-compose ps
```

Once running, access the services:
- **Web Application:** `http://localhost:3000`
- **FastAPI Interactive Docs:** `http://localhost:8000/docs`
- **Readiness Health Probe:** `http://localhost:8000/health/ready`

---

### Running Test Suites

#### Backend Test Suite (Pytest)
```bash
# Execute all 64 automated backend tests with verbose output
pytest backend/tests/ -v
```

#### Frontend Test Suite (Vitest)
```bash
# Execute frontend unit & component tests
npm --prefix frontend test -- --run
```

#### Production Frontend Typecheck & Build
```bash
# Verify TypeScript types and generate production bundle
npm --prefix frontend run build
```

---

## ⚠️ System Limitations & Ethical Disclaimers

1. **Non-Clinical Decision Support Only:** This software is designed for civil engineering, dispatch prioritization, and statistical transportation research. It does **not** provide clinical diagnosis or replace on-scene emergency personnel assessment.
2. **Absence of Real Impact Velocities:** No open census accident database records true physical collision impact velocity (Delta-V). Models approximate crash energy using posted speed limits, road classification, and vehicle mass ratios as proxies.
3. **Jurisdictional Reporting Divergence:** Different countries apply varying legal definitions for road injury severity. In Germany and France, "Serious" injury is legally defined by inpatient hospitalization $\ge 24\text{ hours}$, while the UK and Australia include non-overnight inpatient medical care.
4. **Geographical Sensor Telematics Scope:** The Corridor Sensor Risk model was trained on the TDOT Chattanooga I-24 / I-75 / SR-153 sensor network. It is not calibrated for zero-shot transfer to unmonitored rural roadways or differing sensor grid topologies.

---

## 👥 Authors & Acknowledgments

- **Development & Research:** Global Accident Intelligence Team
- **Public Data Sources:**
  - United Kingdom Department for Transport (DfT)
  - Ministère de l'Intérieur / Observatoire National Interministériel de la Sécurité Routière (France)
  - Statistisches Bundesamt / Destatis (Germany)
  - Department of Transport and Planning, Victoria (Australia)
  - Tennessee Department of Transportation (TDOT)
  - ECMWF Copernicus Climate Change Service (ERA5 Reanalysis)

---

<div align="center">
  <sub>Built with precision for global transportation safety and data-driven infrastructure engineering.</sub>
</div>
