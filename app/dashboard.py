import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide", page_icon="⚡")

st.title("⚡ Energy Consumption Forecasting")
st.markdown("Comparing **ARIMA**, **Prophet**, and **LSTM** models on PJM East hourly energy data (2002–2018)")
st.divider()

@st.cache_data
def load_data():
    arima = pd.read_csv('results/arima_results.csv', index_col=0, parse_dates=True)
    prophet = pd.read_csv('results/prophet_results.csv', index_col=0, parse_dates=True)
    lstm = pd.read_csv('results/lstm_results.csv', index_col=0, parse_dates=True)
    
    arima_m = pd.read_csv('results/arima_metrics.csv')
    prophet_m = pd.read_csv('results/prophet_metrics.csv')
    lstm_m = pd.read_csv('results/lstm_metrics.csv')
    metrics = pd.concat([arima_m, prophet_m, lstm_m], ignore_index=True)
    
    raw = pd.read_csv('data/pjme_clean.csv', index_col='datetime', parse_dates=True)
    
    return arima, prophet, lstm, metrics, raw

arima_r, prophet_r, lstm_r, metrics, raw = load_data()

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

for i, (col, row) in enumerate(zip([col1, col2, col3], metrics.itertuples())):
    with col:
        st.metric(label=row.model, value=f"{row.MAE:.2f} MW", delta=f"MAPE: {row.MAPE:.2f}%", delta_color="inverse")
        st.caption(f"RMSE: {row.RMSE:.2f} MW")

st.divider()

st.subheader("📈 Forecast vs Actual")

selected_models = st.multiselect(
    "Select models to display:",
    ["ARIMA", "Prophet", "LSTM"],
    default=["ARIMA", "Prophet", "LSTM"]
)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(arima_r.index, arima_r['actual'], label='Actual', color='black', linewidth=2, alpha=0.8)

model_data = {
    "ARIMA": (arima_r, '#e74c3c'),
    "Prophet": (prophet_r, '#3498db'),
    "LSTM": (lstm_r, '#2ecc71')
}

for name in selected_models:
    data, color = model_data[name]
    ax.plot(data.index, data['predicted'], label=name, linestyle='--', color=color, alpha=0.8, linewidth=1.5)

ax.set_ylabel('Energy (MW)')
ax.legend(fontsize=11)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

st.subheader("🔍 Raw Data Explorer")

col_left, col_right = st.columns([1, 3])

with col_left:
    year_range = st.slider(
        "Select year range:",
        min_value=int(raw.index.year.min()),
        max_value=int(raw.index.year.max()),
        value=(2016, 2018)
    )
    resample_opt = st.selectbox("Resample:", ["Hourly", "Daily", "Weekly", "Monthly"])

resample_map = {"Hourly": None, "Daily": "D", "Weekly": "W", "Monthly": "ME"}

with col_right:
    filtered = raw[(raw.index.year >= year_range[0]) & (raw.index.year <= year_range[1])]
    
    if resample_map[resample_opt]:
        filtered = filtered.resample(resample_map[resample_opt]).mean()
    
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(filtered.index, filtered['energy_mw'], linewidth=0.8, alpha=0.8)
    ax2.set_title(f'Energy Consumption ({year_range[0]}-{year_range[1]}, {resample_opt})')
    ax2.set_ylabel('Energy (MW)')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.divider()

st.subheader("📉 Error Analysis")

error_model = st.selectbox("Select model for error analysis:", ["ARIMA", "Prophet", "LSTM"])
error_data = {"ARIMA": arima_r, "Prophet": prophet_r, "LSTM": lstm_r}[error_model]

residuals = error_data['actual'] - error_data['predicted']

col_e1, col_e2 = st.columns(2)

with col_e1:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(residuals, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_title(f'{error_model} — Residual Distribution')
    ax3.set_xlabel('Error (MW)')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col_e2:
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.plot(error_data.index, residuals, linewidth=0.8, alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--')
    ax4.set_title(f'{error_model} — Residuals Over Time')
    ax4.set_ylabel('Error (MW)')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

st.divider()

st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        Built with Streamlit · ARIMA · Prophet · LSTM (PyTorch)<br>
        Dataset: PJM Hourly Energy Consumption (Kaggle)
    </div>
    """,
    unsafe_allow_html=True
)