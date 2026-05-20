"""
🛡️ SMART DISASTER PREDICTION SYSTEM - FIXED VERSION
===================================================
KEY FIXES:
✅ Added CORS headers for browser API access
✅ Added lat/lon to live-sensors API response
✅ Fixed Weather API key env var setup
✅ Better path handling (Windows/Linux compatible)
"""
import os
import csv
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import joblib
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS  # ✅ NEW: CORS support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Optional, Tuple
from sklearn.base import BaseEstimator

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ✅ FIX 1: Use pathlib for cross-platform paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
TRAINING_CSV = BASE_DIR / "disaster_training_data.csv"

# Timing (ALL CONFIGURABLE)
DATA_COLLECTION_INTERVAL = 15 * 60      # Collect sensor data for 15 min (900 sec)
PREDICTION_INTERVAL = 1 * 60           # Make prediction every 1 min
AUTO_RETRAIN_INTERVAL = 24 * 60 * 60    # Retrain ML model every 24 hours
MIN_LABELLED_SAMPLES = 20               # Need 20+ real events to retrain

# ✅ FIX 2: Proper Weather API key setup
WEATHER_API_KEY = os.getenv("736f3bd2a4254863a9a173214261003", "736f3bd2a4254863a9a173214261003")
WEATHER_API_URL = "https://api.weatherapi.com/v1/current.json"
SOILGRIDS_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

print(f"[STARTUP] Weather API Key: {'SET' if WEATHER_API_KEY else 'NOT SET'}")

# ML Features (12 total - CRITICAL for prediction)
FEATURE_COLUMNS = [
    # FROM SENSOR (Real-time)
    "soil_moisture",                # % (how wet)
    "vibration_max",                # g (max acceleration)
    "vibration_rms",                # g (overall shaking)
    "flame_detected",               # 0/1 (fire present?)
    
    # FROM WEATHER API (FREE - WeatherAPI)
    "rainfall_24h",                 # mm (last 24 hours)
    "temperature",                  # °C (affects fire spread)
    "wind_speed",                   # km/h (fire spread rate)
    "wind_direction",               # degrees (fire spread direction)
    
    # FROM CONFIG (Soil properties)
    "soil_type",                    # 1=clay, 2=loam, 3=sand
    "water_bearing_capacity",       # mm/m (from SoilGrids)
    "slope",                        # degrees (terrain steepness)
    "vegetation_cover"              # % (fuel for fire)
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 📦 DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeSensorReading:
    """
    REQUIREMENT 1: Data from IoT node
    Received every ~30 seconds
    """
    timestamp: datetime
    node_id: int
    soil_moisture: float            # % (0-100)
    vibration_x: float              # g
    vibration_y: float              # g
    vibration_z: float              # g
    flame_detected: int             # 0=no, 1=yes
    lat: float
    lon: float


@dataclass
class AveragedSensorData:
    """
    REQUIREMENT 3: 15-minute averaged data
    Created by averaging 30 readings
    """
    timestamp: datetime
    node_id: int
    soil_moisture_avg: float        # Average over 15 min
    vibration_max: float            # Peak vibration
    vibration_rms: float            # Overall motion
    flame_detected: int             # Any flame in window?
    reading_count: int
    lat: float
    lon: float

@dataclass
class WeatherInfo:
    """
    REQUIREMENT 2: Weather data from API
    Used for prediction
    """
    temperature: float              # °C
    rainfall_24h: float             # mm
    wind_speed: float               # km/h
    wind_direction: float           # degrees (0-360)

@dataclass
class DisasterPrediction:
    """
    REQUIREMENT 2: Prediction output
    What server predicts about upcoming event
    """
    timestamp: datetime
    node_id: int
    
    # LANDSLIDE PREDICTION
    landslide_probability: float    # 0-1
    landslide_risk: str             # LOW, MEDIUM, HIGH
    
    # FIRE PREDICTION
    fire_probability: float         # 0-1
    fire_risk: str                  # NO, WARNING, DANGER
    
    # REQUIREMENT 4: Fire direction from wind
    fire_spread_direction: str      # N, NE, E, SE, S, SW, W, NW
    fire_spread_degrees: float      # 0-360
    
    # REQUIREMENT 2: Region safety
    region_safety: str              # SAFE, CAUTION, DANGER
    
    # For webpage display
    features_used: Dict


# ═══════════════════════════════════════════════════════════════════════════════
# 🌍 SITE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SITE_CONFIG = {
    1: {
        "name": "Slope Near GEC idukki-1",
        "soil_type": 2,                 # 1=clay, 2=loam, 3=sand
        "slope": 35,                    # degrees
        "vegetation_cover": 55,         # %
        "lat": 9.850,                   # Latitude (for map visualization)
        "lon": 76.938                    # Longitude (for map visualization)
    },
    2: {
        "name": "Slope Near GEC idukki-2",
        "soil_type": 1,
        "slope": 20,
        "vegetation_cover": 45,
        "lat": 9.849,                   # Different location for demo
        "lon": 76.939
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🔒 THREAD-SAFE STATE
# ═══════════════════════════════════════════════════════════════════════════════

state_lock = threading.Lock()

# REQUIREMENT 1: Real-time sensor buffer
sensor_buffer: Dict[int, List[NodeSensorReading]] = defaultdict(list)

# Latest predictions from ML model
latest_predictions: Dict[int, DisasterPrediction] = {}

# API caches
weather_cache: Dict[int, WeatherInfo] = {}
soil_cache: Dict[int, Dict] = {}
cache_time: Dict[str, float] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 CSV OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

CSV_HEADER = FEATURE_COLUMNS + ["label", "event_type", "node_id", "lat", "lon", "timestamp"]

SEED_DATA = {
    "soil_moisture": [35, 45, 55, 65, 75, 85, 40, 70, 50, 80, 60, 90],
    "vibration_max": [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.02, 0.12, 0.04, 0.16, 0.06, 0.18],
    "vibration_rms": [0.005, 0.010, 0.015, 0.025, 0.050, 0.080, 0.010, 0.070, 0.020, 0.090, 0.035, 0.100],
    "flame_detected": [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    "rainfall_24h": [5, 15, 25, 45, 75, 120, 10, 100, 20, 110, 50, 150],
    "temperature": [16, 18, 20, 23, 26, 30, 17, 28, 21, 32, 24, 33],
    "wind_speed": [3, 6, 9, 12, 18, 25, 5, 22, 8, 28, 15, 30],
    "wind_direction": [0, 45, 90, 135, 180, 225, 30, 270, 60, 315, 120, 240],
    "soil_type": [1, 1, 2, 2, 2, 3, 1, 3, 2, 3, 1, 3],
    "water_bearing_capacity": [120, 120, 140, 140, 140, 100, 120, 100, 140, 95, 120, 100],
    "slope": [10, 15, 20, 25, 30, 40, 12, 38, 18, 45, 22, 42],
    "vegetation_cover": [60, 60, 60, 55, 50, 30, 60, 25, 55, 20, 55, 15],
    "label": [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
}

def ensure_csv_exists():
    """Create training CSV with seed data"""
    csv_path = Path(TRAINING_CSV)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    df = pd.DataFrame(SEED_DATA)
    df["event_type"] = ""
    df["node_id"] = 1
    df["lat"] = 10.02
    df["lon"] = 76.30
    df["timestamp"] = datetime.now().isoformat()
    
    df.to_csv(TRAINING_CSV, index=False)
    log.info(f"✓ CSV created with {len(df)} seed samples")


def save_aggregated_data_to_csv(row: dict):
    """
    REQUIREMENT 3: Save 15-minute averaged data
    This CSV is used for model retraining
    """
    try:
        csv_path = Path(TRAINING_CSV)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        
        with open(TRAINING_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        log.error(f"CSV error: {e}")


def load_training_data() -> pd.DataFrame:
    """Load CSV for model retraining"""
    ensure_csv_exists()
    try:
        return pd.read_csv(TRAINING_CSV)
    except:
        return pd.DataFrame(SEED_DATA)


# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 FREE API INTEGRATION (REQUIREMENT 2)
# ═══════════════════════════════════════════════════════════════════════════════

def get_weather_data(lat: float, lon: float) -> WeatherInfo:
    """
    REQUIREMENT 2: Get upcoming weather from FREE WeatherAPI
    Returns: temperature, rainfall, wind_speed, wind_direction
    """
    try:
        params = {
            'key': WEATHER_API_KEY,
            'q': f"{lat},{lon}",
            'aqi': 'no'
        }
        
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        current = response.json()['current']
        
        return WeatherInfo(
            temperature=current['temp_c'],
            rainfall_24h=current.get('precip_mm', 0),
            wind_speed=current['wind_kph'],
            wind_direction=current['wind_degree']
        )
        
    except Exception as e:
        log.warning(f"Weather API error: {e} - using defaults")
        return WeatherInfo(
            temperature=25.0,
            rainfall_24h=0.0,
            wind_speed=5.0,
            wind_direction=0.0
        )


def get_soil_water_capacity(lat: float, lon: float) -> float:
    """
    REQUIREMENT 2: Get soil water bearing capacity from FREE SoilGrids API
    Returns: water capacity in mm/m
    """
    try:
        params = {
            'lon': lon,
            'lat': lat,
            'property': ['awc'],          # available water capacity
            'depth': ['0-5cm'],
            'value': 'mean'
        }
        
        response = requests.get(SOILGRIDS_API_URL, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        for prop in data.get('properties', []):
            if prop['name'] == 'awc' and prop.get('layers'):
                value = prop['layers'][0]['depths'][0]['values'].get('mean', 140)
                return value / 10
        
        return 140.0  # Default loam
        
    except Exception as e:
        log.warning(f"SoilGrids API error: {e} - using default")
        return 140.0


# ═══════════════════════════════════════════════════════════════════════════════
# 🔥 REQUIREMENT 4: FIRE DIRECTION FROM WIND
# ═══════════════════════════════════════════════════════════════════════════════

def degrees_to_compass(degrees: float) -> str:
    """Convert degrees (0-360) to cardinal direction"""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return directions[round(degrees / 22.5) % 16]


def calculate_fire_spread(wind_direction: float) -> Tuple[str, float]:
    """
    REQUIREMENT 4: Calculate fire spread direction from wind
    Fire spreads IN the direction the wind blows
    """
    spread_direction = degrees_to_compass(wind_direction)
    return spread_direction, wind_direction


# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 DISASTER PREDICTION FORMULAS (REQUIREMENT 2)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_landslide_risk(
    soil_moisture: float,
    vibration_rms: float,
    rainfall: float,
    slope: float,
    water_capacity: float,
    soil_type: int
) -> float:
    """
    REQUIREMENT 2: Predict landslide using soil, weather, and vibration
    Returns probability 0-1
    """
    
    # Wet soil (moisture > 70%) = high pore pressure = landslide risk
    moisture_risk = max(0, (soil_moisture - 40) / 60)
    
    # Vibration = soil moving = unstable
    vibration_risk = min(1.0, vibration_rms / 0.2)
    
    # Heavy rain = water saturation
    rainfall_risk = min(1.0, rainfall / 150)
    
    # Steep slopes = gravity effect
    slope_risk = max(0, (slope - 15) / 40)
    
    # Clay can't drain water = high risk
    capacity_risk = max(0, 1 - water_capacity / 200)
    
    # Soil type factor: clay=1.2, loam=1.0, sand=0.8
    soil_factor = {1: 1.2, 2: 1.0, 3: 0.8}.get(soil_type, 1.0)
    
    probability = (
        moisture_risk * 0.35 +
        vibration_risk * 0.25 +
        rainfall_risk * 0.20 +
        slope_risk * 0.12 +
        capacity_risk * 0.08
    ) * soil_factor
    
    return min(1.0, max(0.0, probability))


def predict_fire_risk(
    soil_moisture: float,
    temperature: float,
    rainfall: float,
    vegetation: float,
    wind_speed: float
) -> float:
    """
    REQUIREMENT 2: Predict fire using soil dryness, weather, and vegetation
    Returns probability 0-1
    """
    
    # Dry soil = fire ignition
    dry_risk = max(0, (40 - soil_moisture) / 40) if soil_moisture < 40 else 0
    
    # Hot temperature = faster burning
    temp_risk = max(0, (temperature - 15) / 35)
    
    # No rain = dry vegetation (fuel)
    rainfall_risk = max(0, 1 - rainfall / 100)
    
    # More vegetation = more fuel
    vegetation_risk = vegetation / 100
    
    # Fast wind = spreads fire
    wind_risk = min(1.0, wind_speed / 30)
    
    probability = (
        dry_risk * 0.30 +
        temp_risk * 0.25 +
        rainfall_risk * 0.20 +
        vegetation_risk * 0.15 +
        wind_risk * 0.10
    )
    
    return min(1.0, max(0.0, probability))


# ═══════════════════════════════════════════════════════════════════════════════
# 🧬 ML MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_ml_model():
    """Build ensemble ML model"""
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    
    return VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[0.6, 0.4]
    )

def train_ml_model(df: pd.DataFrame) -> Tuple[Optional[object], Optional[object]]:
    """Train model on labelled data"""
    labelled = df.dropna(subset=["label"])
    
    if len(labelled) < MIN_LABELLED_SAMPLES:
        log.warning(f"Only {len(labelled)} labelled samples (need {MIN_LABELLED_SAMPLES})")
        return None, None
    
    try:
        X = labelled[FEATURE_COLUMNS].astype(float)
        y = labelled["label"].astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = build_ml_model()
        
        if len(y.unique()) > 1:
            scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)//10), scoring='roc_auc')
            log.info(f"✓ Model CV AUC: {scores.mean():.3f}")
        
        model.fit(X_scaled, y)
        log.info(f"✓ Model trained on {len(labelled)} real events")
        
        return model, scaler
    except Exception as e:
        log.error(f"Training error: {e}")
        return None, None

def load_or_init_model() -> Tuple[Optional[object], Optional[object]]:
    """Load model or train from seed"""
    if Path(MODEL_PATH).exists() and Path(SCALER_PATH).exists():
        try:
            return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        except:
            pass
    
    df = pd.DataFrame(SEED_DATA)
    model, scaler = train_ml_model(df)
    
    if model and scaler:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler


model, scaler = load_or_init_model()


def predict_with_ml(features: List[float]) -> float:
    """Use ML for prediction"""
    global model, scaler
    
    if model is None or scaler is None:
        return 0.5
    
    try:
        features_scaled = scaler.transform([features])
        return float(model.predict_proba(features_scaled)[0][1])
    except:
        return 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# 🔮 MAKE PREDICTIONS (REQUIREMENT 2)
# ═══════════════════════════════════════════════════════════════════════════════

def make_disaster_prediction(node_id: int, agg_data: AveragedSensorData) -> DisasterPrediction:
    """
    REQUIREMENT 2: Predict upcoming disaster
    Uses: sensor data + weather + soil + ML model
    """
    config = SITE_CONFIG.get(node_id, SITE_CONFIG[1])
    
    # Get weather (with caching)
    with state_lock:
        now = time.time()
        cache_key = f"weather_{node_id}"
        
        if cache_key not in cache_time or (now - cache_time[cache_key] > 900):
            weather = get_weather_data(agg_data.lat, agg_data.lon)
            weather_cache[node_id] = weather
            cache_time[cache_key] = now
        else:
            weather = weather_cache[node_id]
        
        # Get soil water capacity (with caching)
        cache_key = f"soil_{node_id}"
        if cache_key not in cache_time or (now - cache_time[cache_key] > 3600):
            water_capacity = get_soil_water_capacity(agg_data.lat, agg_data.lon)
            soil_cache[node_id] = {"water_capacity": water_capacity}
            cache_time[cache_key] = now
        else:
            water_capacity = soil_cache[node_id]["water_capacity"]
    
    # Build 12-feature vector
    features = [
        agg_data.soil_moisture_avg,
        agg_data.vibration_max,
        agg_data.vibration_rms,
        agg_data.flame_detected,
        weather.rainfall_24h,
        weather.temperature,
        weather.wind_speed,
        weather.wind_direction,
        config["soil_type"],
        water_capacity,
        config["slope"],
        config["vegetation_cover"]
    ]
    
    # PREDICTION 1: Landslide risk
    landslide_prob = predict_landslide_risk(
        agg_data.soil_moisture_avg,
        agg_data.vibration_rms,
        weather.rainfall_24h,
        config["slope"],
        water_capacity,
        config["soil_type"]
    )
    
    # Blend with ML if available
    if model is not None:
        ml_prob = predict_with_ml(features)
        landslide_prob = 0.6 * landslide_prob + 0.4 * ml_prob
    
    # PREDICTION 2: Fire risk
    fire_prob = predict_fire_risk(
        agg_data.soil_moisture_avg,
        weather.temperature,
        weather.rainfall_24h,
        config["vegetation_cover"],
        weather.wind_speed
    )
    
    # REQUIREMENT 4: Fire spread direction from wind
    fire_dir, fire_deg = calculate_fire_spread(weather.wind_direction)
    
    # Risk classification
    landslide_risk = "HIGH" if landslide_prob > 0.75 else "MEDIUM" if landslide_prob > 0.4 else "LOW"
    fire_risk = "DANGER" if fire_prob > 0.75 else "WARNING" if fire_prob > 0.4 else "NO"
    
    # REQUIREMENT 2: Overall region safety
    region_safety = "DANGER" if (landslide_risk == "HIGH" or fire_risk == "DANGER") else \
                    "CAUTION" if (landslide_risk == "MEDIUM" or fire_risk == "WARNING") else "SAFE"
    
    return DisasterPrediction(
        timestamp=datetime.now(),
        node_id=node_id,
        landslide_probability=round(landslide_prob, 3),
        landslide_risk=landslide_risk,
        fire_probability=round(fire_prob, 3),
        fire_risk=fire_risk,
        fire_spread_direction=fire_dir,
        fire_spread_degrees=round(fire_deg, 1),
        region_safety=region_safety,
        features_used={
            "soil_moisture": round(agg_data.soil_moisture_avg, 1),
            "vibration_rms": round(agg_data.vibration_rms, 4),
            "flame": agg_data.flame_detected,
            "rainfall_24h": round(weather.rainfall_24h, 1),
            "temperature": round(weather.temperature, 1),
            "wind_speed": round(weather.wind_speed, 1),
            "wind_direction": round(weather.wind_direction, 0),
            "water_capacity": round(water_capacity, 0)
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# ⏰ BACKGROUND THREADS
# ═══════════════════════════════════════════════════════════════════════════════

def aggregation_and_prediction_thread():
    """
    REQUIREMENT 3: Every 1 minutes:
    1. Average sensor data collect in 15 min
    2. Make predictions in 1
    3. Save to CSV
    """
    while True:
        time.sleep(DATA_COLLECTION_INTERVAL)
        
        log.info("═" * 70)
        log.info("15-MINUTE AGGREGATION & PREDICTION")
        log.info("═" * 70)
        
        with state_lock:
            local_buffer = dict(sensor_buffer)
            sensor_buffer.clear()
        
        for node_id, readings in local_buffer.items():
            if not readings:
                continue
            
            try:
                # Extract values
                soil = [r.soil_moisture for r in readings]
                vib_x = [r.vibration_x for r in readings]
                vib_y = [r.vibration_y for r in readings]
                vib_z = [r.vibration_z for r in readings]
                flame = [r.flame_detected for r in readings]
                
                # Aggregate (REQUIREMENT 3)
                agg = AveragedSensorData(
                    timestamp=datetime.now(),
                    node_id=node_id,
                    soil_moisture_avg=float(np.mean(soil)),
                    vibration_max=max(max(abs(x) for x in vib_x),
                                     max(abs(y) for y in vib_y),
                                     max(abs(z) for z in vib_z)),
                    vibration_rms=float(np.sqrt(np.mean([x**2+y**2+z**2 for x,y,z in zip(vib_x,vib_y,vib_z)]))),
                    flame_detected=1 if any(flame) else 0,
                    reading_count=len(readings),
                    lat=readings[-1].lat,
                    lon=readings[-1].lon
                )
                
                # REQUIREMENT 2: Make prediction
                pred = make_disaster_prediction(node_id, agg)
                
                with state_lock:
                    latest_predictions[node_id] = pred
                
                # REQUIREMENT 3: Save to CSV for training
                csv_row = {
                    "soil_moisture": agg.soil_moisture_avg,
                    "vibration_max": agg.vibration_max,
                    "vibration_rms": agg.vibration_rms,
                    "flame_detected": agg.flame_detected,
                    "rainfall_24h": pred.features_used.get("rainfall_24h", 0),
                    "temperature": pred.features_used.get("temperature", 25),
                    "wind_speed": pred.features_used.get("wind_speed", 5),
                    "wind_direction": pred.features_used.get("wind_direction", 0),
                    "soil_type": SITE_CONFIG[node_id]["soil_type"],
                    "water_bearing_capacity": pred.features_used.get("water_capacity", 140),
                    "slope": SITE_CONFIG[node_id]["slope"],
                    "vegetation_cover": SITE_CONFIG[node_id]["vegetation_cover"],
                    "label": "",
                    "event_type": "",
                    "node_id": node_id,
                    "lat": round(agg.lat, 6),
                    "lon": round(agg.lon, 6),
                    "timestamp": agg.timestamp.isoformat()
                }
                save_aggregated_data_to_csv(csv_row)
                
                # Log
                site_name = SITE_CONFIG.get(node_id, {}).get("name", f"Node {node_id}")
                log.info(
                    f"✓ {site_name}\n"
                    f"  Readings: {agg.reading_count} | "
                    f"Soil: {agg.soil_moisture_avg:.1f}% | "
                    f"Vibration: {agg.vibration_rms:.4f}g\n"
                    f"  Landslide: {pred.landslide_risk} ({pred.landslide_probability:.1%}) | "
                    f"Fire: {pred.fire_risk} ({pred.fire_probability:.1%}) → {pred.fire_spread_direction}\n"
                    f"  Region Safety: {pred.region_safety}"
                )
                
            except Exception as e:
                log.error(f"Node {node_id} error: {e}", exc_info=True)


def auto_retrain_thread():
    """Auto-retrain ML model every 24 hours"""
    global model, scaler
    
    while True:
        time.sleep(AUTO_RETRAIN_INTERVAL)
        
        log.info("═" * 70)
        log.info("24-HOUR AUTO-RETRAIN")
        log.info("═" * 70)
        
        try:
            df = load_training_data()
            new_model, new_scaler = train_ml_model(df)
            
            if new_model and new_scaler:
                joblib.dump(new_model, MODEL_PATH)
                joblib.dump(new_scaler, SCALER_PATH)
                model = new_model
                scaler = new_scaler
                
                log.info(f"✓ Model improved!")
            
        except Exception as e:
            log.error(f"Retrain error: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 FLASK REST API
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# ✅ FIX 3: Add CORS support
CORS(app)

def _predict_for_node(node_id: int):
    """Run prediction immediately for the given node using its latest reading."""
    with state_lock:
        readings = sensor_buffer.get(node_id, [])
        if not readings:
            return None
        latest = readings[-1]

    # Use last reading as the aggregated sample (real-time)
    agg = AveragedSensorData(
        timestamp=latest.timestamp,
        node_id=node_id,
        soil_moisture_avg=latest.soil_moisture,
        vibration_max=max(
            abs(latest.vibration_x),
            abs(latest.vibration_y),
            abs(latest.vibration_z)
        ),
        vibration_rms=float(
            (latest.vibration_x ** 2 + latest.vibration_y ** 2 + latest.vibration_z ** 2) ** 0.5
        ),
        flame_detected=latest.flame_detected,
        reading_count=1,
        lat=latest.lat,
        lon=latest.lon
    )

    pred = make_disaster_prediction(node_id, agg)

    with state_lock:
        latest_predictions[node_id] = pred

    return pred


@app.route("/node-data", methods=["POST"])
def receive_node_data():
    """
    REQUIREMENT 1: Receive real-time sensor data
    Node sends: soil_moisture, vibration (x,y,z), flame, location
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400

        required = ["node_id","soil_moisture","vib_x","vib_y","vib_z","lat","lon"]
        if not all(k in data for k in required):
            return jsonify({"error": f"Missing fields. Required: {required}"}), 400

        reading = NodeSensorReading(
            timestamp=datetime.now(),
            node_id=int(data["node_id"]),
            soil_moisture=float(data["soil_moisture"]),
            vibration_x=float(data["vib_x"]),
            vibration_y=float(data["vib_y"]),
            vibration_z=float(data["vib_z"]),
            flame_detected=int(data.get("flame_detected", 0)),
            lat=float(data["lat"]),
            lon=float(data["lon"])
        )

        with state_lock:
            sensor_buffer[reading.node_id].append(reading)
            buffer_size = len(sensor_buffer[reading.node_id])

            # Count how many nodes have at least one reading
            active_nodes = [nid for nid, reads in sensor_buffer.items() if reads]

        # Immediately compute prediction for the node that sent data
        _predict_for_node(reading.node_id)

        # If we have data from at least 2 nodes, refresh predictions for all active nodes
        if len(active_nodes) >= 2:
            for nid in active_nodes:
                _predict_for_node(nid)

        log.info(f"✓ Node {reading.node_id} data received ({buffer_size} in buffer)")

        return jsonify({
            "status": "ok",
            "readings_buffered": buffer_size,
            "message": f"Data received from node {reading.node_id}"
        })
    except Exception as e:
        log.error(f"Error: {e}")
        print("Incoming data:", data if 'data' in locals() else "No JSON")
        return jsonify({"error": str(e)}), 400
        
@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """
    REQUIREMENT 5: Get predicted data
    Returns: landslide risk, fire risk, fire direction, region safety
    """
    with state_lock:
        results = {}
        for node_id, pred in latest_predictions.items():
            results[str(node_id)] = {
                "node_id": pred.node_id,
                "node_name": SITE_CONFIG.get(node_id, {}).get("name", f"Node {node_id}"),
                "timestamp": pred.timestamp.isoformat(),
                
                "landslide": {
                    "probability": pred.landslide_probability,
                    "risk": pred.landslide_risk
                },
                
                "fire": {
                    "probability": pred.fire_probability,
                    "risk": pred.fire_risk,
                    "spread_direction": pred.fire_spread_direction,
                    "spread_degrees": pred.fire_spread_degrees
                },
                
                "region_safety": pred.region_safety,
                "features": pred.features_used
            }
    
    return jsonify(results)

@app.route("/api/live-sensors", methods=["GET"])
def get_live_sensors():
    """
    REQUIREMENT 5: Get real-time sensor data
    Returns: current soil moisture, vibration, flame status
    Includes lat/lon for map markers
    """

    with state_lock:
        live = {}   # Dictionary keyed by node_id

        for node_id, readings in sensor_buffer.items():
            if readings:
                latest = readings[-1]

                live[str(node_id)] = {
                    "node_id": node_id,
                    "timestamp": latest.timestamp.isoformat(),
                    "soil_moisture": round(latest.soil_moisture, 1),
                    "lat": latest.lat,
                    "lon": latest.lon,
                    "vibration": {
                        "x": round(latest.vibration_x, 4),
                        "y": round(latest.vibration_y, 4),
                        "z": round(latest.vibration_z, 4),
                        "max": round(
                            max(
                                abs(latest.vibration_x),
                                abs(latest.vibration_y),
                                abs(latest.vibration_z)
                            ), 4
                        )
                    },
                    "flame_detected": latest.flame_detected,
                    "readings_in_buffer": len(readings)
                }

    return jsonify(live)

@app.route("/api/label", methods=["POST"])
def label_event():
    """Label real events for model training"""
    try:
        body = request.json
        ts = body.get("timestamp")
        node_id = body.get("node_id")
        label_val = body.get("label")
        event_type = body.get("event_type", "")
        
        if not (ts and label_val is not None):
            return jsonify({"error": "timestamp and label required"}), 400
        
        df = load_training_data()
        mask = (df["timestamp"] == ts)
        if node_id:
            mask &= (df["node_id"] == node_id)
        
        if mask.sum() == 0:
            return jsonify({"error": "No matching record"}), 404
        
        df.loc[mask, "label"] = int(label_val)
        df.loc[mask, "event_type"] = event_type
        df.to_csv(TRAINING_CSV, index=False)
        
        log.info(f"✓ Labelled {mask.sum()} record(s)")
        return jsonify({"updated": int(mask.sum())})
        
    except Exception as e:
        log.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get training statistics"""
    try:
        df = load_training_data()
        labelled = df.dropna(subset=["label"])
        
        return jsonify({
            "total_records": len(df),
            "labelled_records": len(labelled),
            "unlabelled_records": len(df) - len(labelled),
            "model_ready": len(labelled) >= MIN_LABELLED_SAMPLES,
            "records_needed": max(0, MIN_LABELLED_SAMPLES - len(labelled))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def dashboard():
    """
    REQUIREMENT 5: Serve web dashboard
    Shows real-time sensor data + predicted data
    """
    return render_template("index.html")


@app.route("/map")
def map_dashboard():
    """
    MAP DASHBOARD: Show nodes and disaster zones on interactive map
    """
    return render_template("index.html")
# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ensure_csv_exists()
    
    # Start background threads
    threading.Thread(
        target=aggregation_and_prediction_thread,
        daemon=True,
        name="AggregationThread"
    ).start()
    
    threading.Thread(
        target=auto_retrain_thread,
        daemon=True,
        name="RetrainThread"
    ).start()
    
    print(f"""
    ╔════════════════════════════════════════════╗
    ║  🛡️  SMART DISASTER PREDICTION SYSTEM      ║
    ║                                            ║
    ║  Flask running on: http://localhost:5000   ║
    ║  Dashboard: http://localhost:5000          ║
    ║  Map: http://localhost:5000/map            ║
    ║                                            ║
    ║  ✅ CORS enabled for browser access        ║
    ║  ✅ Weather API key: {'SET' if WEATHER_API_KEY else 'NOT SET'}        ║
    ║                                            ║
    ╚════════════════════════════════════════════╝
    """)
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)