import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import sqlite3
from influxdb_client import InfluxDBClient
import time
from requests.exceptions import ReadTimeout

# Streamlit page configuration
st.set_page_config(page_title="DNS Anomaly Detection Dashboard", layout="wide")

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "attacks" not in st.session_state:
    st.session_state.attacks = []

# API endpoint
API_URL = "https://mizzony-dns-anomalies-detection.hf.space/predict"

# InfluxDB configuration
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "6gjE97dCC24hgOgWNmRXPqOS0pfc0pMSYeh5psL8e5u2T8jGeV1F17CU-U1z05if0jfTEmPRW9twNPSXN09SRQ=="
INFLUXDB_ORG = "Anormally Detection"
INFLUXDB_BUCKET = "realtime_dns"

# SQLite database for attack logging
DB_PATH = "attacks.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attacks
                 (timestamp TEXT, inter_arrival_time REAL, dns_rate REAL, request_rate REAL,
                  reconstruction_error REAL, anomaly INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# Query InfluxDB for DNS data with retry
def get_dns_data(range_start="-30m", retries=3, delay=5):
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=30000)  # 30 seconds
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r._measurement == "dns")
      |> filter(fn: (r) => r._field == "inter_arrival_time" or r._field == "dns_rate")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    '''
    for attempt in range(retries):
        try:
            result = client.query_api().query(query)
            if result and len(result) > 0:
                for record in result[0].records:
                    return {
                        "timestamp": record.get_time().strftime("%Y-%m-%d %H:%M:%S"),
                        "inter_arrival_time": record.values.get("inter_arrival_time", None),
                        "dns_rate": record.values.get("dns_rate", None)
                    }
            return None
        except (ReadTimeout, Exception) as e:
            if attempt < retries - 1:
                st.warning(f"InfluxDB query attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"InfluxDB query failed after {retries} attempts: {e}")
                return None
        finally:
            client.close()

# Query InfluxDB for historical data with retry
def get_historical_dns_data(start_time, end_time, retries=3, delay=5):
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=30000)
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r._measurement == "dns")
      |> filter(fn: (r) => r._field == "inter_arrival_time" or r._field == "dns_rate")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: false)
    '''
    for attempt in range(retries):
        try:
            result = client.query_api().query(query)
            data = []
            if result and len(result) > 0:
                for record in result[0].records:
                    data.append({
                        "timestamp": record.get_time().strftime("%Y-%m-%d %H:%M:%S"),
                        "inter_arrival_time": record.values.get("inter_arrival_time", None),
                        "dns_rate": record.values.get("dns_rate", None)
                    })
            return pd.DataFrame(data) if data else pd.DataFrame(columns=["timestamp", "inter_arrival_time", "dns_rate"])
        except (ReadTimeout, Exception) as e:
            if attempt < retries - 1:
                st.warning(f"Historical InfluxDB query attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"Historical InfluxDB query failed after {retries} attempts: {e}")
                return pd.DataFrame(columns=["timestamp", "inter_arrival_time", "dns_rate"])
        finally:
            client.close()

# Title and description
st.title("DNS Anomaly Detection Dashboard")
st.markdown("""
Monitor DNS traffic in real-time using InfluxDB data, detecting anomalies with a pre-trained autoencoder model.
Analyze live and historical data, capture attacks, and evaluate model performance with Grafana-style visualizations.
""")

# Sidebar for controls
st.sidebar.header("Dashboard Controls")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 30 min", "Last 1 hour", "Last 24 hours", "Last 7 days", "Last 14 days", "Last 30 days"],
    index=4  # Default to "Last 14 days"
)
threshold = st.sidebar.slider("Anomaly Threshold", 0.01, 1.0, 0.1, 0.01)
enable_alerts = st.sidebar.checkbox("Enable Attack Alerts", value=True)

# Convert time range to InfluxDB format
time_ranges = {
    "Last 30 min": ("-30m", "now()"),
    "Last 1 hour": ("-1h", "now()"),
    "Last 24 hours": ("-24h", "now()"),
    "Last 7 days": ("-7d", "now()"),
    "Last 14 days": ("-14d", "now()"),
    "Last 30 days": ("-30d", "now()")
}
start_time, end_time = time_ranges[time_range]

# Auto-refresh every 30 minutes (1800000 ms)
st_autorefresh(interval=1800000, key="datarefresh")

# Manual input
st.header("Manual Input")
col1, col2 = st.columns(2)
with col1:
    inter_arrival_time = st.number_input(
        "Inter Arrival Time (seconds)", min_value=0.001, max_value=100.0, value=0.01938, step=0.001
    )
with col2:
    dns_rate = st.number_input(
        "DNS Rate", min_value=0.0, max_value=100.0, value=2.0, step=0.1
    )

if st.button("Detect Anomaly"):
    payload = {"inter_arrival_time": inter_arrival_time, "dns_rate": dns_rate}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.predictions.append(result)
        st.session_state.predictions = st.session_state.predictions[-1000:]
        if result["anomaly"] == 1:
            st.session_state.attacks.append(result)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''INSERT INTO attacks (timestamp, inter_arrival_time, dns_rate, request_rate,
                        reconstruction_error, anomaly) VALUES (?, ?, ?, ?, ?, ?)''',
                      (result["timestamp"], result["inter_arrival_time"], result["dns_rate"],
                       result["request_rate"], result["reconstruction_error"], result["anomaly"]))
            conn.commit()
            conn.close()
            if enable_alerts:
                st.error(f"Attack Detected! Timestamp: {result['timestamp']}, Error: {result['reconstruction_error']:.6f}")
        st.success("Prediction successful!")
        st.json(result)
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")

# Real-time monitoring
st.header("Real-Time Monitoring")
if st.checkbox("Enable Live Stream", value=True):
    try:
        data = get_dns_data()
        if data and data["inter_arrival_time"] is not None and data["dns_rate"] is not None:
            payload = {
                "inter_arrival_time": data["inter_arrival_time"],
                "dns_rate": data["dns_rate"]
            }
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            result["timestamp"] = data["timestamp"]
            st.session_state.predictions.append(result)
            st.session_state.predictions = st.session_state.predictions[-1000:]
            if result["anomaly"] == 1:
                st.session_state.attacks.append(result)
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('''INSERT INTO attacks (timestamp, inter_arrival_time, dns_rate, request_rate,
                            reconstruction_error, anomaly) VALUES (?, ?, ?, ?, ?, ?)''',
                          (result["timestamp"], result["inter_arrival_time"], result["dns_rate"],
                           result["request_rate"], result["reconstruction_error"], result["anomaly"]))
                conn.commit()
                conn.close()
                if enable_alerts:
                    st.error(f"Attack Detected! Timestamp: {result['timestamp']}, Error: {result['reconstruction_error']:.6f}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error in live stream: {e}")

# Load historical data
st.header("Attack Analysis")
historical_df = get_historical_dns_data(start_time, end_time)

# Process historical data through API
if not historical_df.empty:
    historical_predictions = []
    for _, row in historical_df.iterrows():
        if pd.notnull(row["inter_arrival_time"]) and pd.notnull(row["dns_rate"]):
            payload = {
                "inter_arrival_time": float(row["inter_arrival_time"]),
                "dns_rate": float(row["dns_rate"])
            }
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                result["timestamp"] = row["timestamp"]
                historical_predictions.append(result)
                if result["anomaly"] == 1:
                    st.session_state.attacks.append(result)
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute('''INSERT INTO attacks (timestamp, inter_arrival_time, dns_rate, request_rate,
                                reconstruction_error, anomaly) VALUES (?, ?, ?, ?, ?, ?)''',
                              (result["timestamp"], result["inter_arrival_time"], result["dns_rate"],
                               result["request_rate"], result["reconstruction_error"], result["anomaly"]))
                    conn.commit()
                    conn.close()
            except requests.exceptions.RequestException:
                pass
    if historical_predictions:
        st.session_state.predictions.extend(historical_predictions)
        st.session_state.predictions = st.session_state.predictions[-1000:]

# Display predictions
if st.session_state.predictions:
    df = pd.DataFrame(st.session_state.predictions)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Table
    st.subheader("Recent Predictions")
    st.dataframe(df[["timestamp", "inter_arrival_time", "dns_rate", "request_rate", "reconstruction_error", "anomaly"]])
    
    # Time-series plot
    st.subheader("Time-Series Analysis")
    fig_time = px.line(
        df,
        x="timestamp",
        y=["reconstruction_error", "inter_arrival_time", "dns_rate"],
        title="DNS Metrics Over Time",
        color_discrete_map={"reconstruction_error": "red", "inter_arrival_time": "blue", "dns_rate": "green"}
    )
    fig_time.add_hline(y=threshold, line_dash="dash", line_color="black", annotation_text=f"Threshold ({threshold})")
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Heatmap
    st.subheader("Reconstruction Error Heatmap")
    df["hour"] = df["timestamp"].dt.hour
    heatmap_data = df.pivot_table(values="reconstruction_error", index="hour", aggfunc="mean")
    fig_heatmap = px.imshow(
        heatmap_data.T,
        labels=dict(x="Hour of Day", y="Metric", color="Reconstruction Error"),
        title="Average Reconstruction Error by Hour"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Pie chart
    st.subheader("Anomaly Distribution")
    anomaly_counts = df["anomaly"].value_counts().reset_index()
    anomaly_counts.columns = ["Anomaly", "Count"]
    anomaly_counts["Anomaly"] = anomaly_counts["Anomaly"].map({0: "Normal", 1: "Attack"})
    fig_pie = px.pie(
        anomaly_counts,
        names="Anomaly",
        values="Count",
        title="Normal vs. Attack Distribution",
        color="Anomaly",
        color_discrete_map={"Normal": "blue", "Attack": "red"}
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Attack details
    if st.session_state.attacks:
        st.subheader("Attack Details")
        attacks_df = pd.DataFrame(st.session_state.attacks)
        attacks_df["timestamp"] = pd.to_datetime(attacks_df["timestamp"])
        st.dataframe(attacks_df[["timestamp", "inter_arrival_time", "dns_rate", "reconstruction_error"]])
    
    # Model performance
    st.subheader("Model Performance")
    st.write("**Threshold**: ", threshold)
    fig_hist = px.histogram(
        df,
        x="reconstruction_error",
        color="anomaly",
        title="Reconstruction Error Distribution",
        color_discrete_map={0: "blue", 1: "red"},
        nbins=50
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Summary metrics
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(df))
    col2.metric("Attack Rate", f"{df['anomaly'].mean():.2%}")
    col3.metric("Recent Attacks", df.tail(10)["anomaly"].sum())
    
    # Export data
    st.subheader("Export Data")
    csv = df.to_csv(index=False)
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    if st.session_state.attacks:
        attacks_csv = pd.DataFrame(st.session_state.attacks).to_csv(index=False)
        st.download_button("Download Attacks", attacks_csv, "attacks.csv", "text/csv")
else:
    st.info("No predictions yet. Enable live stream or use manual input.")
