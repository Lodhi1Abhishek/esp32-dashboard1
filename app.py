from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import pandas as pd
import joblib
import requests
import eventlet
from datetime import datetime, timedelta
import os

# Initialize App
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# 1. LOAD AI MODEL
try:
    model = joblib.load('landslide_model.pkl')
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    model = None

# --- HELPER: Fetch Live Satellite Data ---
def get_ai_features(lat, lon):
    try:
        print(f"🔍 Fetching features for: {lat}, {lon}")
        
        # A. RAIN (Live & Recent Past)
        # We use the Forecast API because it has "past_days" which gives us immediate history
        url_rain = "https://api.open-meteo.com/v1/forecast"
        params_rain = {
            'latitude': lat, 'longitude': lon,
            'daily': 'precipitation_sum', 
            'timezone': 'auto',
            'past_days': 7,      # Get last 7 days
            'forecast_days': 1   # Get today
        }
        r_rain = requests.get(url_rain, params=params_rain, timeout=5).json()
        
        # Sum the last 7 days of rain
        if 'daily' in r_rain and 'precipitation_sum' in r_rain['daily']:
            # Take the first 7 values (the past days)
            rain_values = r_rain['daily']['precipitation_sum']
            rain_7day = sum(rain_values[:7]) 
        else:
            rain_7day = 0.0

        # B. ELEVATION & SLOPE
        url_elev = "https://api.open-meteo.com/v1/elevation"
        r_elev = requests.get(url_elev, params={'latitude': [lat, lat+0.001], 'longitude': [lon, lon]}, timeout=5).json()
        
        if 'elevation' in r_elev:
            elevs = r_elev['elevation']
            elevation = elevs[0]
            slope = abs(elevs[0] - elevs[1]) * 100 # Proxy calculation
        else:
            elevation = 0
            slope = 0

        # C. SOIL (ISRIC)
        url_soil = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params_soil = {'lat': lat, 'lon': lon, 'property': ['clay', 'sand', 'silt'], 'depth': '0-5cm', 'value': 'mean'}
        r_soil = requests.get(url_soil, params=params_soil, timeout=5).json()
        
        # Default values if soil API fails (e.g. ocean)
        sand, silt, clay = 0, 0, 0
        
        if 'properties' in r_soil and 'layers' in r_soil['properties']:
            vals = {l['name']: l['depths'][0]['values']['mean']/10.0 for l in r_soil['properties']['layers']}
            sand = vals.get('sand', 0)
            silt = vals.get('silt', 0)
            clay = vals.get('clay', 0)

        # DEBUG: Print what we found
        print(f"   -> Data Found: Rain={rain_7day}mm, Slope={slope}, Clay={clay}%")

        # D. RETURN ORDERED DATA
        # CRITICAL: This list MUST match the training order exactly!
        # Training Order: ['rain_7day', 'elevation', 'slope', 'sand', 'silt', 'clay']
        features = pd.DataFrame([{
            'rain_7day': rain_7day, 
            'elevation': elevation, 
            'slope': slope,
            'sand': sand, 
            'silt': silt, 
            'clay': clay
        }])
        
        # Reorder columns explicitly to be safe
        features = features[['rain_7day', 'elevation', 'slope', 'sand', 'silt', 'clay']]
        
        return features

    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

# --- ROUTE 1: Dashboard HTML ---
@app.route('/')
def index():
    return render_template('index.html')

# --- ROUTE 2: ESP32 Data Endpoint ---
@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json
    
    # 1. Process AI Prediction
    if model and data.get('base_lat', 0) != 0:
        features_df = get_ai_features(data['base_lat'], data['base_lng'])
        if features_df is not None:
            # Predict
            risk = int(model.predict_proba(features_df)[0][1] * 100)
            data['landslide_risk'] = risk
            print(f"   -> Calculated Risk: {risk}%")
        else:
            print("   -> Feature fetch failed, risk = 0")
    
    # 2. Send to Dashboard
    socketio.emit('sensor_update', data)
    return "Data Received", 200

# --- ROUTE 3: Manual Check (Map Click) ---
@app.route('/api/check-location', methods=['POST'])
def check_location():
    data = request.json
    lat, lon = data.get('lat'), data.get('lon')
    
    if not model: return jsonify({"error": "AI Model Offline"}), 500

    features_df = get_ai_features(lat, lon)
    if features_df is not None:
        risk = int(model.predict_proba(features_df)[0][1] * 100)
        
        # Convert DataFrame back to dict for the frontend JSON
        # We take the first row (iloc[0]) and convert to dict
        features_dict = features_df.iloc[0].to_dict()
        
        return jsonify({
            "risk_score": risk, 
            "features": features_dict
        })
    
    return jsonify({"error": "Analysis Failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, host='0.0.0.0', port=port)
