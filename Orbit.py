import pandas as pd
import re
from orbit.models import DLT
from orbit.diagnostics.backtest import BackTester
from orbit.diagnostics.metrics import smape
from orbit.forecaster import Forecaster

# Function to sanitize product keys for filenames
def sanitize_filename(filename):
    invalid_chars = r'[\/:*?"<>|]'
    sanitized = re.sub(invalid_chars, '_', filename)
    return sanitized

# Load datasets
Keys = pd.read_csv('Data/Materialeliste.csv', delimiter=';', encoding='ISO-8859-1')
df = pd.read_csv('Data/All_Forecast_Data.csv', delimiter=';', encoding='ISO-8859-1')

# Convert 'week-year' to 'YYYY-MM-DD'
df['ds'] = df['ds'].apply(lambda x: pd.to_datetime(f"{x}-1", format='%W-%Y-%w'))

evaluation_results = []

# Iterate through each product
for product_key in Keys.iloc[:, 0]:
    product_data = df[df['Keys'] == product_key].copy()

    if product_data.empty:
        print(f"No data found for {product_key}")
        continue

    # Replace missing values in 'y' with 0
    product_data['y'] = product_data['y'].fillna(0)

    # Count the number of data points where sales ('y') are greater than 0
    num_nonzero_points = (product_data['y'] > 0).sum()

    # If the product has fewer than 20 non-zero data points, skip the product
    if num_nonzero_points < 20:
        print(f"Not enough non-zero data points for product {product_key}")
        continue

    split_point = int(len(product_data) * 0.8)
    train_df = product_data.iloc[:split_point]  # Initial training data (first 80%)
    test_df = product_data.iloc[split_point:]   # Remaining 20% for testing

    # Rolling forecast for each week in the test set (N-3)
    rolling_forecasts = []
    for i in range(len(test_df)):
        # Use all data up to the current test week minus 2 weeks for training (N-3)
        if i < 3:
            print(f"Skipping forecast for {product_key} as we have less than 3 test points")
            continue

        current_train_df = pd.concat([train_df, test_df.iloc[:i-3]])

        # Initialize DLT model from Orbit
        model = DLT(
            response_col='y',
            date_col='ds',
            seasonality=52,  # Weekly seasonality (adjust based on your dataset's frequency)
            seed=42
        )

        # Fit model with the training data
        model.fit(df=current_train_df)

        # Forecast for the next week
        future_df = pd.DataFrame({'ds': [test_df.iloc[i]['ds']]})
        forecaster = Forecaster(model=model)
        forecast = forecaster.predict(future_df)

        # Prepare forecast output
        forecast['Actual sales'] = test_df.iloc[i]['y']
        forecast['product'] = product_key

        # Ensure non-negative forecast values
        forecast['prediction'] = forecast['prediction'].clip(lower=0)

        rolling_forecasts.append(forecast)

    # Combine rolling forecasts into a single DataFrame
    forecast_df = pd.concat(rolling_forecasts)

    # Save forecast to CSV
    sanitized_key = sanitize_filename(product_key)
    forecast_file = f"Data/Future_Forecast(N-3)/{sanitized_key}_forecast.csv"
    forecast_df.to_csv(forecast_file, index=False)

    # Evaluation metrics (similar to Prophet evaluation)
    forecast_df['Difference'] = forecast_df['prediction'] - forecast_df['Actual sales']
    forecast_df['Absolut'] = forecast_df['Difference'].abs()
    forecast_df['Weight'] = forecast_df[['prediction', 'Actual sales']].max(axis=1)
    forecast_df['WMAPE'] = (1 - forecast_df['Absolut'] / forecast_df['Weight']) * 100
    forecast_df['NormalizedBias'] = (forecast_df['prediction'] - forecast_df['Actual sales']) / (forecast_df['prediction'] + forecast_df['Actual sales']) * 100
    
    # Calculate overall WMAPE and Normalized Bias for the product
    TotalForecast = forecast_df['prediction'].sum()
    TotalActual = forecast_df['Actual sales'].sum()
    TotalWeight = forecast_df['Weight'].sum()
    TotalABS = forecast_df['Absolut'].sum()
    
    TotalWMAPE = (1 - TotalABS / TotalWeight) * 100
    TotalNormalizedBias = (TotalForecast - TotalActual) / (TotalForecast + TotalActual) * 100
    
    evaluation_results.append({
        'Product': sanitized_key,
        'TotalNormalizedBias': TotalNormalizedBias,
        'TotalWMAPE': TotalWMAPE
    })

# Save evaluation results
evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df.to_csv('Data/Evaluation_Results(N-3).csv', index=False)
print("All forecasts have been saved.")
