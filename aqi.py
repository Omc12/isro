import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import requests
import datetime
from datetime import timedelta
import time
import sys
import subprocess


try:
    from xgboost import XGBRegressor
    print("XGBoost already installed.")
except ImportError:
    print("Installing XGBoost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    from xgboost import XGBRegressor


API_KEY = "579b464db66ec23bdd0000010619fb37307244fc7255d0a82bc40152"
WAQI_API_URL = "https://api.waqi.info/feed/"

class AQIForecastModel:
 
    def __init__(self, forecast_horizon=1, model_type='rf'):
        
        self.forecast_horizon = forecast_horizon
        self.feature_columns = None
        self.target_column = 'aqi'
        self.model_type = model_type
        
    def _prepare_features(self, df):
    
        data = df.copy()
        

        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data.set_index('date', inplace=True)
            else:
                raise ValueError("DataFrame must have a 'date' column or datetime index")
        

        data['month'] = data.index.month
        data['day_of_week'] = data.index.dayofweek
        data['day_of_year'] = data.index.dayofyear
        data['is_weekend'] = data.index.dayofweek >= 5
        

        for col in ['aqi', 'temperature', 'humidity', 'wind_speed', 'pm25']:
            if col in data.columns:
                for lag in range(1, 8):  # 1 to 7 days lag
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
                
                # Create rolling statistics
                data[f'{col}_roll_mean_3'] = data[col].rolling(window=3).mean()
                data[f'{col}_roll_mean_7'] = data[col].rolling(window=7).mean()
                data[f'{col}_roll_std_7'] = data[col].rolling(window=7).std()
                

        data[f'aqi_next_{self.forecast_horizon}'] = data['aqi'].shift(-self.forecast_horizon)
        
        return data
    
    def train(self, data, target_col=None, test_size=0.2, random_state=42):
        
        print("Preparing features...")
        df = self._prepare_features(data)
        

        if target_col is None:
            target_col = f'aqi_next_{self.forecast_horizon}'
        

        df = df.dropna()
        

        self.feature_columns = [
            # Lag features
            'aqi_lag_1', 'aqi_lag_2', 'aqi_lag_3',
            'temperature_lag_1', 'humidity_lag_1', 'wind_speed_lag_1',
            'pm25_lag_1',
            
            # Rolling statistics
            'aqi_roll_mean_3', 'aqi_roll_mean_7',
            'temperature_roll_mean_3', 'humidity_roll_mean_3',
            
            # Time-based features
            'month', 'day_of_week', 'is_weekend'
        ]
        

        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        self.target_column = target_col
        

        X = df[self.feature_columns]
        y = df[self.target_column]
        
        print(f"Training with {len(X)} samples and {len(self.feature_columns)} features")
        

        if self.model_type == 'xgb':
            self.pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state
                ))
            ])
        else:  # Default to RandomForest
            self.pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state
                ))
            ])
        

        tscv = TimeSeriesSplit(n_splits=5)
        scores = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        print("Performing time-series cross-validation...")
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            

            self.pipeline.fit(X_train, y_train)
            

            y_pred = self.pipeline.predict(X_test)
            

            from sklearn.metrics import mean_absolute_error
            scores['mae'].append(mean_absolute_error(y_test, y_pred))
            scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            scores['r2'].append(r2_score(y_test, y_pred))
        

        print("Training final model on all data...")
        self.pipeline.fit(X, y)
        

        if hasattr(self.pipeline.named_steps['model'], 'feature_importances_'):
            self.feature_importances = dict(zip(
                self.feature_columns,
                self.pipeline.named_steps['model'].feature_importances_
            ))
            

            self.feature_importances = {
                k: v for k, v in sorted(
                    self.feature_importances.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )
            }
            
            print("\nTop 5 important features:")
            for i, (feature, importance) in enumerate(list(self.feature_importances.items())[:5]):
                print(f"{i+1}. {feature}: {importance:.4f}")
        else:
            self.feature_importances = {}
        
        print("Model training complete.")
        print(f"Average MAE: {np.mean(scores['mae']):.2f}")
        print(f"Average RMSE: {np.mean(scores['rmse']):.2f}")
        print(f"Average R²: {np.mean(scores['r2']):.2f}")
        
        return {
            'mae': np.mean(scores['mae']),
            'rmse': np.mean(scores['rmse']),
            'r2': np.mean(scores['r2']),
            'feature_importances': self.feature_importances
        }
    
    def predict(self, features):
        """
        Make AQI predictions using the trained model.
        
        Parameters:
        -----------
        features : dict or pandas.DataFrame
            Features to use for prediction
            
        Returns:
        --------
        float or numpy.ndarray
            Predicted AQI value(s)
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call 'train()' first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure all required features are present
        missing_features = [col for col in self.feature_columns if col not in features.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract the features in the correct order
        X = features[self.feature_columns]
        
        # Make prediction
        return self.pipeline.predict(X)
    
    def forecast_next_days(self, last_data, days=7):
        
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call 'train()' first.")
        
        # Make a copy of the last data
        forecast_data = last_data.copy()
        results = []
        
        current_date = forecast_data.index.max()
        
        for i in range(1, days + 1):
            # Prepare features for the next day
            next_date = current_date + pd.Timedelta(days=i)
            features = self._prepare_forecast_features(forecast_data, next_date)
            
            # Make prediction
            predicted_aqi = self.predict(features)[0]
            
            # Store result
            results.append({
                'date': next_date,
                'aqi': predicted_aqi
            })
            
            # Update forecast data with the new prediction for recursive forecasting
            new_row = features.copy()
            new_row['aqi'] = predicted_aqi
            new_row.index = [next_date]
            
            forecast_data = pd.concat([forecast_data, new_row])
        
        return pd.DataFrame(results).set_index('date')
    
    def _prepare_forecast_features(self, historical_data, forecast_date):
        features = pd.DataFrame(index=[forecast_date])
        
        # Add time-based features
        features['month'] = forecast_date.month
        features['day_of_week'] = forecast_date.dayofweek
        features['day_of_year'] = forecast_date.dayofyear
        features['is_weekend'] = forecast_date.dayofweek >= 5
        
        # Add lag features from historical data
        for lag in range(1, 8):
            lag_date = forecast_date - pd.Timedelta(days=lag)
            if lag_date in historical_data.index:
                for col in ['aqi', 'temperature', 'humidity', 'wind_speed', 'pm25']:
                    if col in historical_data.columns:
                        features[f'{col}_lag_{lag}'] = historical_data.loc[lag_date, col]
        
        # Add rolling statistics
        for window in [3, 7]:
            for col in ['aqi', 'temperature', 'humidity', 'wind_speed', 'pm25']:
                if col in historical_data.columns:
                    # Get data for the window period before forecast date
                    window_start = forecast_date - pd.Timedelta(days=window)
                    window_data = historical_data.loc[
                        (historical_data.index >= window_start) & 
                        (historical_data.index < forecast_date)
                    ]
                    
                    # Calculate statistics if enough data is available
                    if len(window_data) > 0:
                        features[f'{col}_roll_mean_{window}'] = window_data[col].mean()
                        if window == 7:  # Only calculate std for 7-day window
                            features[f'{col}_roll_std_{window}'] = window_data[col].std()
        
        return features
    
    def plot_forecast(self, historical_data, forecast_data):
        """
        Plot historical AQI data and forecasted values.
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            DataFrame with historical AQI data
        forecast_data : pandas.DataFrame
            DataFrame with forecasted AQI values
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plotted figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data['aqi'], 
                label='Historical AQI', color='blue', marker='o')
        
        # Plot forecasted data
        ax.plot(forecast_data.index, forecast_data['aqi'], 
                label='Forecasted AQI', color='red', marker='x', linestyle='--')
        
        # Add vertical line separating historical and forecasted data
        ax.axvline(x=historical_data.index.max(), color='black', linestyle=':')
        
        # Customize the plot
        ax.set_title('AQI Forecast', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('AQI', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add AQI category bands
        self._add_aqi_bands(ax)
        
        plt.tight_layout()
        return fig
    
    def _add_aqi_bands(self, ax):
        """Add colored bands for AQI categories to a plot."""
        # Define AQI categories and colors
        aqi_bands = [
            (0, 50, 'green', 'Good'),
            (51, 100, 'yellow', 'Moderate'),
            (101, 150, 'orange', 'Unhealthy for Sensitive Groups'),
            (151, 200, 'red', 'Unhealthy'),
            (201, 300, 'purple', 'Very Unhealthy'),
            (301, 500, 'maroon', 'Hazardous')
        ]
        
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        
        for lower, upper, color, label in aqi_bands:
            if lower < ylim[1] and upper > ylim[0]:
                effective_lower = max(lower, ylim[0])
                effective_upper = min(upper, ylim[1])
                ax.axhspan(effective_lower, effective_upper, 
                           color=color, alpha=0.2, label=label)
        
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.pipeline is None:
            raise ValueError("No trained model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'feature_importances': self.feature_importances,
            'forecast_horizon': self.forecast_horizon,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        model = AQIForecastModel(
            forecast_horizon=model_data['forecast_horizon'],
            model_type=model_data.get('model_type', 'rf')
        )
        model.pipeline = model_data['pipeline']
        model.feature_columns = model_data['feature_columns']
        model.target_column = model_data['target_column']
        model.feature_importances = model_data['feature_importances']
        
        return model


def fetch_real_time_aqi(city, api_key=API_KEY):
    try:
        url = f"{WAQI_API_URL}/{city}/?token={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == 'ok':
            aqi_value = data['data']['aqi']
            
            # Extract additional data if available
            iaqi = data['data'].get('iaqi', {})
            
            result = {
                'aqi': aqi_value,
                'time': datetime.datetime.now(),
                'city': city
            }
            
            # Add individual pollutants if available
            pollutant_map = {
                'pm25': 'pm25',
                'pm10': 'pm10',
                'o3': 'o3',
                'no2': 'no2',
                'so2': 'so2',
                'co': 'co',
                't': 'temperature',
                'h': 'humidity',
                'w': 'wind_speed'
            }
            
            for api_key, our_key in pollutant_map.items():
                if api_key in iaqi:
                    result[our_key] = iaqi[api_key]['v']
            
            return result
        else:
            print(f"Error fetching data for {city}: {data.get('data')}")
            return None
    except Exception as e:
        print(f"Error fetching real-time AQI data: {str(e)}")
        return None


def fetch_historical_aqi(city, days=50, api_key=API_KEY):
    print(f"Fetching {days} days of historical AQI data for {city}...")
    
    # First, try to load from CSV files if available
    try:
        # Look for data files that might contain this city
        city_files = []
        for file in os.listdir("data"):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join("data", file))
                if 'City' in df.columns and city.lower() in [c.lower() for c in df['City'].unique()]:
                    city_files.append(file)
        
        if city_files:
            print(f"Found {len(city_files)} data files containing {city}")
            all_data = []
            
            for file in city_files:
                df = pd.read_csv(os.path.join("data", file))
                # Filter for the requested city
                if 'City' in df.columns:
                    city_data = df[df['City'].str.lower() == city.lower()]
                    if len(city_data) > 0:
                        all_data.append(city_data)
            
            if all_data:
                combined_data = pd.concat(all_data)
                
                # Process the data into the format we need
                if 'From Date' in combined_data.columns:
                    combined_data['date'] = pd.to_datetime(combined_data['From Date'])
                elif 'date' in combined_data.columns:
                    combined_data['date'] = pd.to_datetime(combined_data['date'])
                
                # Rename columns to our standard names
                column_mapping = {
                    'AQI': 'aqi',
                    'PM2.5': 'pm25',
                    'PM10': 'pm10',
                    'Temperature': 'temperature',
                    'Humidity': 'humidity',
                    'Wind Speed': 'wind_speed'
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in combined_data.columns:
                        combined_data[new_col] = combined_data[old_col]
                
                # Set the date as index
                combined_data.set_index('date', inplace=True)
                
                # Sort by date
                combined_data = combined_data.sort_index()
                
                # Take only the last 'days' days
                if len(combined_data) > days:
                    combined_data = combined_data.iloc[-days:]
                
                print(f"Loaded {len(combined_data)} historical records from local files")
                return combined_data
    except Exception as e:
        print(f"Error loading data from local files: {str(e)}")
    
    # If we couldn't load from files or there wasn't enough data,
    # fetch from the API
    print("Fetching historical data from API...")
    
    # For the WAQI API, we can only get current data
    # So we'll use a workaround to simulate historical data
    current_data = fetch_real_time_aqi(city, api_key)
    
    if current_data is None:
        print(f"Could not fetch data for {city}. Using synthetic data.")
        # Generate synthetic data as a fallback
        return create_synthetic_data(days)
    
    # Create a synthetic historical dataset based on the current reading
    # with realistic variations
    base_aqi = current_data['aqi']
    base_temp = current_data.get('temperature', 25)
    base_humidity = current_data.get('humidity', 60)
    base_wind = current_data.get('wind_speed', 10)
    base_pm25 = current_data.get('pm25', base_aqi / 4.0)
    
    dates = [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(days, 0, -1)]
    
    # Create a DataFrame with some realistic variations
    data = {
        'date': dates,
        'aqi': [max(10, base_aqi + np.random.normal(0, base_aqi * 0.15)) for _ in range(days)],
        'pm25': [max(5, base_pm25 + np.random.normal(0, base_pm25 * 0.15)) for _ in range(days)],
        'temperature': [max(5, base_temp + np.random.normal(0, 5)) for _ in range(days)],
        'humidity': [min(max(20, base_humidity + np.random.normal(0, 10)), 100) for _ in range(days)],
        'wind_speed': [max(0, base_wind + np.random.normal(0, 3)) for _ in range(days)]
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Add seasonal patterns
    days_array = np.arange(days)
    seasonal_effect = 20 * np.sin(2 * np.pi * days_array / 30)  # Monthly cycle
    weekly_effect = 15 * np.sin(2 * np.pi * days_array / 7)     # Weekly cycle
    
    # Apply the patterns with some noise
    df['aqi'] = df['aqi'] + seasonal_effect + weekly_effect + np.random.normal(0, 10, size=days)
    
    # Make sure AQI stays within reasonable bounds
    df['aqi'] = df['aqi'].clip(10, 500)
    
    print(f"Generated {days} days of historical-like data")
    return df


def create_synthetic_data(days=365):
    """
    Generates a synthetic dataset for AQI forecasting when real data is not available.
    """
    print(f"Generating synthetic data for {days} days...")
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]

    np.random.seed(42)
    temperature = np.random.uniform(15, 35, size=days) - 5 * np.cos(2 * np.pi * np.arange(days) / 365.25)
    humidity = np.random.uniform(40, 80, size=days)
    wind_speed = np.random.uniform(5, 25, size=days)
    
    # Add another key pollutant: PM2.5, with seasonal and random effects
    pm25_base = 20 + (temperature - 25) * 1.5 - (wind_speed - 10) * 2.5 + 15 * np.sin(2 * np.pi * np.arange(days) / 180)
    pm25 = pm25_base + np.random.normal(0, 8, size=days)
    pm25 = np.maximum(5, pm25)  # PM2.5 is rarely zero
    
    # Make AQI dependent on PM2.5 and other meteorological factors
    seasonal_effect = 25 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    aqi = (
        50 
        + pm25 * 1.8  # AQI is strongly influenced by PM2.5
        + (humidity - 60) * 0.5  # Higher humidity can trap pollutants
        - (wind_speed - 10) * 3  # Wind helps disperse pollutants
        + seasonal_effect  # Seasonal variation
        + np.random.normal(0, 15, size=days)  # Random variations
    )
    
    # Ensure AQI is within realistic bounds
    aqi = np.clip(aqi, 20, 500)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'aqi': aqi,
        'pm25': pm25,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
    })
    
    # Set date as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    return df


def get_health_advisory(aqi):
    """Return health advisory based on AQI value"""
    if aqi <= 50:
        return {
            "level": "Good",
            "color": "green",
            "advice": "Air quality is considered satisfactory, and air pollution poses little or no risk."
        }
    elif aqi <= 100:
        return {
            "level": "Moderate",
            "color": "yellow",
            "advice": "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people."
        }
    elif aqi <= 150:
        return {
            "level": "Unhealthy for Sensitive Groups",
            "color": "orange",
            "advice": "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
        }
    elif aqi <= 200:
        return {
            "level": "Unhealthy",
            "color": "red",
            "advice": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
        }
    elif aqi <= 300:
        return {
            "level": "Very Unhealthy",
            "color": "purple",
            "advice": "Health warnings of emergency conditions. The entire population is more likely to be affected."
        }
    else:
        return {
            "level": "Hazardous",
            "color": "maroon",
            "advice": "Health alert: everyone may experience more serious health effects."
        }


def train_city_model(city, model_type='xgb'):
    """
    Train an AQI forecasting model for a specific city using recent historical data.
    
    Parameters:
    -----------
    city : str
        City name to train model for
    model_type : str
        Type of model to use ('rf' for Random Forest, 'xgb' for XGBoost)
        
    Returns:
    --------
    AQIForecastModel
        Trained model for the specified city
    """
    print(f"Training AQI forecasting model for {city}...")
    
    # Fetch historical data for training
    historical_data = fetch_historical_aqi(city, days=50)
    
    # Create and train the model
    model = AQIForecastModel(forecast_horizon=3, model_type=model_type)
    metrics = model.train(historical_data)
    
    # Save the model
    safe_city = "".join(c for c in city if c.isalnum())
    model_path = f"models/{safe_city}_model.joblib"
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)
    
    print(f"Model for {city} trained and saved to {model_path}")
    return model


def forecast_city_aqi(city):
    """
    Generate AQI forecasts for a specific city using real-time data and historical trends.
    
    Parameters:
    -----------
    city : str
        City name to forecast AQI for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with AQI forecasts for the next 7 days
    matplotlib.figure.Figure
        Plot of the forecasts
    str
        Path to the saved forecast image
    """
    print(f"Generating AQI forecast for {city}...")
    
    # Fetch current AQI data from the WAQI API
    current_data = fetch_real_time_aqi(city)
    
    if current_data:
        print(f"Current AQI for {city}: {current_data['aqi']}")
    else:
        print(f"Could not fetch current AQI for {city}. Using only historical data.")
    
    # Fetch historical data for the past 50 days
    historical_data = fetch_historical_aqi(city, days=50)
    
    # Check if we have a pre-trained model for this city
    safe_city = "".join(c for c in city if c.isalnum())
    model_path = f"models/{safe_city}_model.joblib"
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model for {city}...")
        model = AQIForecastModel.load_model(model_path)
    else:
        print(f"No pre-trained model found for {city}. Training a new model...")
        model = train_city_model(city)
    
    # If we have current data, update the historical data with it
    if current_data:
        today = pd.DataFrame({
            'aqi': [current_data['aqi']],
            'pm25': [current_data.get('pm25', current_data['aqi'] / 4)],
            'temperature': [current_data.get('temperature', 25)],
            'humidity': [current_data.get('humidity', 60)],
            'wind_speed': [current_data.get('wind_speed', 10)]
        }, index=[pd.Timestamp.now().floor('D')])
        
        # Add today's data to the historical data
        historical_data = pd.concat([historical_data, today])
    
    # Generate forecasts for the next 7 days
    forecasts = model.forecast_next_days(historical_data, days=7)
    
    # Generate plot
    fig = model.plot_forecast(historical_data.iloc[-30:], forecasts)
    
    # Update plot title
    fig.axes[0].set_title(f'AQI Forecast for {city}', fontsize=15)
    
    # Create a DataFrame with forecast information and health recommendations
    forecast_results = []
    for date, row in forecasts.iterrows():
        aqi_value = row['aqi']
        health_info = get_health_advisory(aqi_value)
        
        forecast_results.append({
            'date': date.strftime('%Y-%m-%d'),
            'aqi': round(aqi_value, 1),
            'category': health_info['level'],
            'color': health_info['color'],
            'health_advice': health_info['advice']
        })
    
    # Save the plot
    os.makedirs("output", exist_ok=True)
    plot_path = f"output/{safe_city}_forecast.png"
    fig.savefig(plot_path)
    
    return pd.DataFrame(forecast_results), fig, plot_path


def interactive_city_forecast():
    """
    Interactive command-line interface for getting city AQI forecasts.
    """
    print("\n" + "=" * 50)
    print("Real-Time AQI Forecasting Tool")
    print("=" * 50)
    print("\nThis tool uses real-time AQI data and historical trends to forecast air quality.")
    
    popular_cities = [
        "Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad",
        "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Vadodara", "Bhopal",
        "Beijing", "Shanghai", "Bangkok", "London", "Paris", "New York", "Los Angeles"
    ]
    
    while True:
        print("\nOptions:")
        print("1. Select from popular cities")
        print("2. Enter a custom city name")
        print("3. Quit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "3":
            break
        
        if choice == "1":
            print("\nPopular Cities:")
            for i, city in enumerate(popular_cities, 1):
                print(f"{i}. {city}")
            
            city_idx = input("\nSelect city number: ")
            try:
                idx = int(city_idx) - 1
                if 0 <= idx < len(popular_cities):
                    selected_city = popular_cities[idx]
                else:
                    print("Invalid selection.")
                    continue
            except ValueError:
                print("Please enter a valid number.")
                continue
        elif choice == "2":
            selected_city = input("\nEnter city name: ")
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            continue
        
        try:
            print(f"\nFetching real-time AQI data and generating forecast for {selected_city}...")
            forecasts, fig, plot_path = forecast_city_aqi(selected_city)
            
            print("\nAQI Forecast for", selected_city)
            print("-" * 30)
            for _, row in forecasts.iterrows():
                print(f"Date: {row['date']}, AQI: {row['aqi']}, Category: {row['category']} ({row['color']})")
                
            print(f"\nForecast plot saved to: {plot_path}")
            print("\nHealth Advisory:")
            for _, row in forecasts.iterrows():
                print(f"- For {row['date']} ({row['category']}): {row['health_advice']}")
                
            # Ask if user wants to display the plot
            show_plot = input("\nShow forecast plot? (y/n): ")
            if show_plot.lower() == 'y':
                plt.figure(fig.number)
                plt.show()
                
        except Exception as e:
            print(f"Error generating forecast for {selected_city}: {str(e)}")
    
    print("\nThank you for using the AQI Forecasting Tool!")


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Start the interactive interface
    interactive_city_forecast()