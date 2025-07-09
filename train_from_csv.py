import pandas as pd
from aqi import AQIForecastModel

# Load the CSV
df = pd.read_csv('3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (1).csv')

# Parse the date
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')

# Pivot so each row is a timestamped observation for a station
pivoted = df.pivot_table(
    index=['last_update', 'country', 'state', 'city', 'station', 'latitude', 'longitude'],
    columns='pollutant_id',
    values='pollutant_avg'
).reset_index()

# Rename columns for model compatibility
pivoted = pivoted.rename(columns={
    'last_update': 'date',
    'PM2.5': 'pm25',
    'PM10': 'pm10',
    'NO2': 'no2',
    'SO2': 'so2',
    'CO': 'co',
    'OZONE': 'ozone',
    # Add more if needed
})

# For AQI, you may need to compute or use one of the pollutants as a proxy (e.g., pm25 or pm10)
# Here, we'll use pm25 as a proxy for AQI (you can adjust as needed)
pivoted['aqi'] = pivoted['pm25']

# Drop rows with missing AQI
pivoted = pivoted.dropna(subset=['aqi'])

# Set date as index
pivoted = pivoted.set_index('date')

# Select only the columns needed for the model
model_df = pivoted[['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'ozone']].copy()

# Fill missing values (optional, model will also impute)
model_df = model_df.fillna(method='ffill').fillna(method='bfill')

# Train the model
model = AQIForecastModel(forecast_horizon=3, model_type='xgb')
model.train(model_df)

# Save the model
model.save_model('models/custom_csv_model.joblib')
print("Model trained and saved as models/custom_csv_model.joblib")