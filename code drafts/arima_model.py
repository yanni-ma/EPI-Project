from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

def run_arima(county_data, order=(1, 1, 0)):
    if not isinstance(county_data.index, pd.DatetimeIndex):
        raise ValueError("The data index must be a DateTimeIndex.")
    
    county_data = county_data.resample('W').sum()
    

    if len(county_data) < 20: 
        raise ValueError("Not enough data for ARIMA modeling after aggregation. Ensure sufficient weekly data.")
    

    train_size = int(len(county_data) * 0.8)
    train, test = county_data[:train_size], county_data[train_size:]
    
    
    if train['new_cases'].std() == 0:
        raise ValueError("Insufficient variability in new_cases for ARIMA modeling.")
    
    
    print("Fitting ARIMA model...")
    model = ARIMA(train['new_cases'], order=order)
    model_fit = model.fit()
    
    
    forecast = model_fit.forecast(steps=len(test))
    
    
    mse = mean_squared_error(test['new_cases'], forecast)
    print(f"Mean Squared Error: {mse}")
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(train['new_cases'], label='Training Data')
    plt.plot(test['new_cases'], label='Testing Data')
    plt.plot(test.index, forecast, label='Forecast', linestyle='dashed')
    plt.title('ARIMA Forecast (Weekly Data)')
    plt.xlabel('Date')
    plt.ylabel('Weekly New Cases')
    plt.legend()
    plt.show()
    
    return model_fit, mse
