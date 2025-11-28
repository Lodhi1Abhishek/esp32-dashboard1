from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import pandas as pd
import joblib
import requests
import eventlet
from datetime import datetime, timedelta

# Initialize App
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# 1. LOAD AI MODEL
try:
    model = joblib.load('landslide_model.pkl')
    print("✅ AI Model Loaded Successfully")
except:
    print("❌ Warning: 'landslide_model.pkl' not found. AI features will be disabled.")
    model = None

# --- HELPER: Fetch Satellite Data for AI ---
def get_ai_features(lat, lon):
    try:
        # A. RAIN (Last 7 Days from Open-Meteo)
        today = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url_rain = "https://archive-api.open-meteo.com/v1/archive"
        params_rain = {'latitude': lat, 'longitude': lon, 'start_date': start, 'end_date': today, 'daily': 'precipitation_sum'}
        r_rain = requests.get(url_rain, params=params_rain).json()
        rain_7day = sum(r_rain.get('daily', {}).get('precipitation_sum', [0]))

        # B. ELEVATION & SLOPE
        url_elev = "https://api.open-meteo.com/v1/elevation"
        r_elev = requests.get(url_elev, params={'latitude': [lat, lat+0.001], 'longitude': [lon, lon]}).json()
        elevs = r_elev.get('elevation', [0, 0])
        elevation = elevs[0]
        slope = abs(elevs[0] - elevs[1]) * 100 # Proxy for slope

        # C. SOIL (ISRIC)
        url_soil = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params_soil = {'lat': lat, 'lon': lon, 'property': ['clay', 'sand', 'silt'], 'depth': '0-5cm', 'value': 'mean'}
        r_soil = requests.get(url_soil, params=params_soil).json()
        vals = {l['name']: l['depths'][0]['values']['mean']/10.0 for l in r_soil['properties']['layers']}

        # Prepare DataFrame for Model
        features = pd.DataFrame([{
            'rain_7day': rain_7day, 
            'elevation': elevation, 
            'slope': slope,
            'sand': vals.get('sand', 30), 
            'silt': vals.get('silt', 30), 
            'clay': vals.get('clay', 30)
        }])
        
        return features
    except Exception as e:
        print(f"API Error: {e}")
        return None

# --- ROUTE 1: Dashboard HTML ---
@app.route('/')
def index():
    return render_template('index.html')

# --- ROUTE 2: ESP32 Data Endpoint ---
@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json
    print(f"📡 Received from ESP32: {data}")

    # 1. Process AI Prediction if GPS is valid
    ai_risk = 0
    if model and data.get('base_lat', 0) != 0:
        features = get_ai_features(data['base_lat'], data['base_lng'])
        if features is not None:
            # Predict Probability (0 to 1) -> Convert to %
            ai_risk = int(model.predict_proba(features)[0][1] * 100)
    
    # 2. Add Risk to Data Packet
    data['landslide_risk'] = ai_risk
    
    # 3. Send to Dashboard via WebSocket
    socketio.emit('sensor_update', data)
    
    return "Data Received", 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)