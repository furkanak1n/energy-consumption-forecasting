from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class EnergyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(EnergyLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.squeeze()

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Predict energy consumption using LSTM model trained on PJM East data (2002-2018)",
    version="1.0.0"
)

WINDOW_SIZE = 30

df = pd.read_csv('data/pjme_clean.csv', index_col='datetime', parse_dates=True)
df_daily = df.resample('D').mean()

scaler = MinMaxScaler()
scaler.fit(df_daily[df_daily.index < '2018-01-01'])

device = torch.device('cpu')
model = EnergyLSTM().to(device)
model.load_state_dict(torch.load('models/lstm_model.pth', map_location=device))
model.eval()

arima_m = pd.read_csv('results/arima_metrics.csv')
prophet_m = pd.read_csv('results/prophet_metrics.csv')
lstm_m = pd.read_csv('results/lstm_metrics.csv')
all_metrics = pd.concat([arima_m, prophet_m, lstm_m], ignore_index=True)

class ForecastRequest(BaseModel):
    days: int = 7
    
    class Config:
        json_schema_extra = {
            "example": {"days": 7}
        }

class ForecastResponse(BaseModel):
    model: str
    forecast_days: int
    predictions: list[dict]

@app.get("/")
def root():
    return {
        "project": "Energy Consumption Forecasting",
        "dataset": "PJM East Hourly (2002-2018)",
        "models": ["ARIMA", "Prophet", "LSTM (PyTorch)"],
        "endpoints": {
            "/forecast": "POST - Get energy consumption forecast",
            "/metrics": "GET - Model performance comparison",
            "/health": "GET - API health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True, "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
def get_metrics():
    results = []
    for _, row in all_metrics.iterrows():
        results.append({
            "model": row["model"],
            "MAE": round(row["MAE"], 2),
            "RMSE": round(row["RMSE"], 2),
            "MAPE": round(row["MAPE"], 2)
        })
    return {"metrics": results}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    if request.days < 1 or request.days > 90:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 90")
    
    last_data = df_daily['energy_mw'].values[-WINDOW_SIZE:]
    scaled = scaler.transform(last_data.reshape(-1, 1))
    
    predictions = []
    current_window = scaled.flatten().tolist()
    
    for i in range(request.days):
        input_seq = np.array(current_window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        input_tensor = torch.FloatTensor(input_seq).to(device)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()
        
        pred_value = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        
        last_date = df_daily.index[-1] + pd.Timedelta(days=i + 1)
        predictions.append({
            "date": last_date.strftime("%Y-%m-%d"),
            "predicted_mw": round(float(pred_value), 2)
        })
        
        current_window.append(float(pred_scaled))
    
    return ForecastResponse(
        model="LSTM (PyTorch)",
        forecast_days=request.days,
        predictions=predictions
    )