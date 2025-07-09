from flask import Flask, request, jsonify
from flask_cors import CORS
from aqi import AQIForecastModel, get_health_advisory
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Always load the custom model
MODEL_PATH = 'models/custom_csv_model.joblib'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Custom model not found at {MODEL_PATH}. Please train and save it first.")
model = AQIForecastModel.load_model(MODEL_PATH)

@app.route('/api/aqi/current', methods=['GET'])
def get_current_aqi():
    try:
        df = pd.read_csv('3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (1).csv')
        df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
        pivoted = df.pivot_table(
            index=['last_update', 'country', 'state', 'city', 'station', 'latitude', 'longitude'],
            columns='pollutant_id',
            values='pollutant_avg'
        ).reset_index()
        pivoted = pivoted.rename(columns={
            'last_update': 'date',
            'PM2.5': 'pm25',
            'PM10': 'pm10',
            'NO2': 'no2',
            'SO2': 'so2',
            'CO': 'co',
            'OZONE': 'ozone',
        })
        # Use the first available pollutant value as AQI
        pivoted['aqi'] = pivoted[['pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']].bfill(axis=1).iloc[:, 0]
        pivoted = pivoted.dropna(subset=['aqi'])
        pivoted = pivoted.set_index('date')
        model_df = pivoted[['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']].copy()
        model_df = model_df.fillna(method='ffill').fillna(method='bfill')
        last_row = model_df.iloc[-1]
        return jsonify({
            'aqi': last_row['aqi'],
            'pm25': last_row['pm25'],
            'pm10': last_row['pm10'],
            'no2': last_row['no2'],
            'so2': last_row['so2'],
            'co': last_row['co'],
            'ozone': last_row['ozone'],
            'timestamp': str(last_row.name)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/aqi/forecast', methods=['GET'])
def get_aqi_forecast():
    try:
        df = pd.read_csv('3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (1).csv')
        df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
        pivoted = df.pivot_table(
            index=['last_update', 'country', 'state', 'city', 'station', 'latitude', 'longitude'],
            columns='pollutant_id',
            values='pollutant_avg'
        ).reset_index()
        pivoted = pivoted.rename(columns={
            'last_update': 'date',
            'PM2.5': 'pm25',
            'PM10': 'pm10',
            'NO2': 'no2',
            'SO2': 'so2',
            'CO': 'co',
            'OZONE': 'ozone',
        })
        pivoted['aqi'] = pivoted[['pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']].bfill(axis=1).iloc[:, 0]
        pivoted = pivoted.dropna(subset=['aqi'])
        pivoted = pivoted.set_index('date')
        model_df = pivoted[['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']].copy()
        model_df = model_df.fillna(method='ffill').fillna(method='bfill')
        n = min(30, len(model_df))
        historical_data = model_df.iloc[-n:]
        print('--- DEBUG: historical_data shape:', historical_data.shape)
        print(historical_data.head())
        try:
            forecasts = model.forecast_next_days(historical_data, days=7)
        except Exception as pred_err:
            print('--- DEBUG: Forecasting error:', pred_err)
            return jsonify({'error': f'Forecasting error: {pred_err}'}), 500
        results = []
        for date, row in forecasts.iterrows():
            aqi_value = row['aqi']
            health_info = get_health_advisory(aqi_value)
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'aqi': round(aqi_value, 1),
                'category': health_info['level'],
                'color': health_info['color'],
                'health_advice': health_info['advice']
            })
        print('--- DEBUG: forecast results:', results)
        return jsonify(results)
    except Exception as e:
        print('--- DEBUG: General error in /api/aqi/forecast:', e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/aqi/alerts', methods=['GET'])
def get_health_alerts():
    # Use the last AQI value from the training data
    try:
        df = pd.read_csv('3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (1).csv')
        df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
        pivoted = df.pivot_table(
            index=['last_update', 'country', 'state', 'city', 'station', 'latitude', 'longitude'],
            columns='pollutant_id',
            values='pollutant_avg'
        ).reset_index()
        pivoted = pivoted.rename(columns={
            'last_update': 'date',
            'PM2.5': 'pm25',
            'PM10': 'pm10',
            'NO2': 'no2',
            'SO2': 'so2',
            'CO': 'co',
            'OZONE': 'ozone',
        })
        pivoted['aqi'] = pivoted['pm25']
        pivoted = pivoted.dropna(subset=['aqi'])
        pivoted = pivoted.set_index('date')
        model_df = pivoted[['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']].copy()
        model_df = model_df.fillna(method='ffill').fillna(method='bfill')
        last_aqi = model_df['aqi'].iloc[-1]
        advisory = get_health_advisory(last_aqi)
        return jsonify({'aqi': last_aqi, 'advisory': advisory})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/aqi/heatmap', methods=['GET'])
def get_heatmap():
    try:
        df = pd.read_csv('3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (1).csv')
        df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
        pivoted = df.pivot_table(
            index=['last_update', 'country', 'state', 'city', 'station', 'latitude', 'longitude'],
            columns='pollutant_id',
            values='pollutant_avg'
        ).reset_index()
        pivoted = pivoted.rename(columns={
            'last_update': 'date',
            'PM2.5': 'pm25',
            'PM10': 'pm10',
            'NO2': 'no2',
            'SO2': 'so2',
            'CO': 'co',
            'OZONE': 'ozone',
        })
        pivoted['aqi'] = pivoted[['pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']].bfill(axis=1).iloc[:, 0]
        pivoted = pivoted.dropna(subset=['aqi'])
        # For each unique (city, station), get the latest record
        latest_points = pivoted.sort_values('date').groupby(['city', 'station', 'latitude', 'longitude'], as_index=False).last()
        points = []
        for _, row in latest_points.iterrows():
            points.append({
                'city': row['city'],
                'station': row['station'],
                'aqi': row['aqi'],
                'lat': row['latitude'],
                'lon': row['longitude']
            })
        return jsonify({'points': points})
    except Exception as e:
        return jsonify({'points': [], 'error': str(e)}), 500

@app.route('/api/aqi/subscribe', methods=['POST'])
def subscribe_push():
    data = request.json
    return jsonify({'status': 'subscribed', 'data': data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10042, debug=True) 