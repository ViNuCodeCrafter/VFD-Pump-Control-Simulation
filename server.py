from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
import uvicorn
import io

app = FastAPI(title="VFD Water Pressure ML Backend")

# --- THINGSBOARD CONFIG ---
TB_HOST = "thingsboard.cloud"
TB_TOKEN = "WCt536zZpj09FfxLm1Ir"

def send_telemetry_to_cloud(data: dict):
    url = f"https://{TB_HOST}/api/v1/{TB_TOKEN}/telemetry"
    try:
        requests.post(url, json=data, timeout=3)
    except Exception as e:
        print(f"ThingsBoard Sync Failed: {e}")

# Enable CORS for the frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL MODEL STATE ---
model = None
encoder = LabelEncoder()
feature_cols = ['floor_no', 'time_slot_encoded', 'current_pressure', 'flow_rate', 'tank_level', 'people_count', 'height_from_ground', 'previous_usage']
target_cols = ['required_pressure', 'required_flow_rate', 'pump_speed', 'valve_command', 'water_allocated']

# --- DATA LOADING OR GENERATION ---
def get_dataset():
    csv_path = "water_pressure_vfd_dataset_100rows_integer.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        print(f"CSV not found, generating synthetic unique data. Error: {e}")
        data = []
        time_slots = ['Morning', 'Afternoon', 'Evening', 'Night']
        for i in range(100):
            # Ensuring non-repetition via random noise
            floor = (i % 6) + 1
            ts = time_slots[(i // 25) % 4]
            height = floor * 3
            # Add small random variations to avoid exact duplicates
            people = int(10 + (floor * 2) + np.random.randint(0, 10))
            usage = int(40 + np.random.randint(0, 20))
            
            # Simulated outputs with slight stochasticity
            req_p = int(np.ceil(1.5 + (height * 0.1) + (people * 0.05) + np.random.uniform(-0.1, 0.1)))
            req_f = 20 + floor + (10 if ts == 'Morning' else 0) + np.random.randint(-2, 3)
            speed = min(100, 40 + (floor * 8) + (people * 1.5) + np.random.randint(-3, 4))
            
            data.append({
                'floor_no': floor, 'time_slot': ts, 'current_pressure': 1.0,
                'flow_rate': 20 - (floor * 2), 'tank_level': 90 - (i % 20),
                'people_count': people, 'avg_usage_per_person': usage,
                'height_from_ground': height, 'previous_usage': 350 + (i * 10) + np.random.randint(0, 50),
                'valve_status': 1 if ts != 'Night' else 0,
                'required_pressure': req_p, 'required_flow_rate': max(10, req_f),
                'pump_speed': max(20, speed), 'valve_command': 1 if ts != 'Night' else 0, 
                'water_allocated': (people * usage) + (floor * 50)
            })
        return pd.DataFrame(data)

# --- DATA SCHEMA ---
class PredictRequest(BaseModel):
    floor_no: int
    time_slot: str
    current_pressure: float
    flow_rate: float
    tank_level: float
    people_count: int
    avg_usage_per_person: float
    height_from_ground: float
    previous_usage: float
    valve_status: int

@app.on_event("startup")
async def startup_event():
    global model, encoder
    df = get_dataset()
    
    # Feature columns based on CSV
    input_features = ['floor_no', 'current_pressure', 'flow_rate', 'tank_level', 'people_count', 'avg_usage_per_person', 'height_from_ground', 'previous_usage', 'valve_status']
    
    # Encode time_slot
    df['time_slot_encoded'] = encoder.fit_transform(df['time_slot'])
    
    X = df[['time_slot_encoded'] + input_features]
    y = df[['required_pressure', 'required_flow_rate', 'pump_speed', 'valve_command', 'water_allocated']]
    
    # Train actual Random Forest
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    print("Backend Model Trained with Unique Dataset.")

# --- ENDPOINTS ---
@app.get("/health")
async def health():
    return {"status": "online", "model_trained": model is not None}

# --- PREDICTION LOGGING ---
LOG_FILE = "Output_Prediction_100rows_integer.csv"

def log_prediction(data: dict):
    # Check if file exists to write header
    import os
    write_header = not os.path.exists(LOG_FILE)
    
    # Get current row count for row#
    row_count = 0
    if not write_header:
        with open(LOG_FILE, 'r') as f:
            row_count = sum(1 for _ in f) - 1 # Subtract header
            
    df_log = pd.DataFrame([data])
    df_log.insert(0, 'row#', row_count + 1)
    
    df_log.to_csv(LOG_FILE, mode='a', index=False, header=write_header)

@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained")
    
    try:
        # Prepare input matching the X columns in startup_event
        ts_encoded = encoder.transform([req.time_slot])[0]
        input_data = [[
            ts_encoded, req.floor_no, req.current_pressure, req.flow_rate,
            req.tank_level, req.people_count, req.avg_usage_per_person,
            req.height_from_ground, req.previous_usage, req.valve_status
        ]]
        
        # Predict
        preds = model.predict(input_data)[0]
        pump_speed = round(preds[2], 2)
        
        # --- DIGITAL TWIN FEEDBACK ---
        from digital_twin import PumpDigitalTwin
        twin = PumpDigitalTwin()
        simulation = twin.simulate(pump_speed)
        
        result = {
            "floor": req.floor_no,
            "time_slot": req.time_slot,
            "required_pressure_bar": round(preds[0], 2),
            "required_flow_rate_lpm": round(preds[1], 2),
            "pump_speed_percent": pump_speed,
            "valve_command": "ON" if preds[3] > 0.5 else "OFF",
            "predicted_water_amount_liters": int(preds[4]),
            "digital_twin": simulation
        }
        
        # Log to CSV
        log_prediction(result)
        
        # --- CLOUD SYNC ---
        cloud_payload = {
            "req_pressure": result['required_pressure_bar'],
            "pump_speed": pump_speed,
            "sim_pressure": simulation['simulated_pressure_bar'],
            "energy_kw": simulation['energy_consumption_kw'],
            "valve": result['valve_command']
        }
        send_telemetry_to_cloud(cloud_payload)
        
        return {
            **result,
            "analysis": f"AI predicts {result['required_pressure_bar']} bar. Digital Twin simulation confirms a resulting hydraulic pressure of {simulation['simulated_pressure_bar']} bar with {simulation['energy_consumption_kw']}kW consumption."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
