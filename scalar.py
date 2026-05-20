from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 CONFIGURATION - Use forward slashes!
# ═══════════════════════════════════════════════════════════════════════════════

SCALER_PATH = "./scaler.pkl"
CSV_PATH = "d:/workspace/Projects/Mini project/disaster_training_data.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# Create or load scaler
# ═══════════════════════════════════════════════════════════════════════════════

if Path(SCALER_PATH).exists():
    print("✓ Loading existing scaler...")
    scaler = joblib.load(SCALER_PATH)
else:
    print("✓ Creating new scaler from CSV data...")
    
    # Step 1: Load data from CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ Loaded {len(df)} rows from CSV")
        
        # Step 2: Select only numeric columns (ignore label, event_type, etc)
        feature_columns = [
            "soil_moisture", "vibration_max", "vibration_rms", "flame_detected",
            "rainfall_24h", "temperature", "wind_speed", "wind_direction",
            "soil_type", "water_bearing_capacity", "slope", "vegetation_cover"
        ]
        
        X = df[feature_columns].astype(float)
        print(f"✓ Selected {len(feature_columns)} features")
        
        # Step 3: Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(X)
        print("✓ Scaler fitted successfully!")
        
        # Step 4: Save for future use
        joblib.dump(scaler, SCALER_PATH)
        print(f"✓ Scaler saved to {SCALER_PATH}")
        
    except FileNotFoundError:
        print(f"❌ CSV file not found: {CSV_PATH}")
        print("   Using seed data instead...")
        
        # Create sample data if CSV doesn't exist
        import numpy as np
        sample_data = np.random.randn(100, 12)
        scaler = StandardScaler()
        scaler.fit(sample_data)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✓ Scaler created from sample data and saved")

print("\n✅ Done! Scaler is ready to use.")
