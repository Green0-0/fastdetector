import streamlit as st
import requests
import os
import time
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="API Spend Tracker", page_icon="💸", layout="wide")
st.title("💸 LiteLLM API Spend Tracker")
st.write("Monitoring `OPENAI_API_KEY` spend in real-time to detect unauthorized usage.")

# Load env variables from .env if present
if "env_loaded" not in st.session_state:
    try:
        with open(".env") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v.strip('\'"')
    except Exception:
        pass
    st.session_state.env_loaded = True

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    st.error("No OPENAI_API_KEY found in .env or environment.")
    st.stop()

# Sidebar controls
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", min_value=1, max_value=60, value=3)
if st.sidebar.button("Clear History"):
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "Spend ($)"])

# Initialize history dataframe
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "Spend ($)"])

try:
    # Fetch data
    resp = requests.get("https://litellm-01.oit.duke.edu/key/info", headers={"Authorization": f"Bearer {API_KEY}"})
    if resp.status_code == 200:
        data = resp.json()
        spend = data.get("info", {}).get("spend", 0.0)
        
        # Append to history
        now = datetime.now()
        new_row = pd.DataFrame({"Timestamp": [now], "Spend ($)": [spend]})
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        
        df = st.session_state.history
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Spend", f"${spend:.6f}")
        col2.metric("Max Budget", f"${data.get('info', {}).get('max_budget', 0.0):.2f}")
        
        if len(df) > 1:
            spend_diff = df.iloc[-1]["Spend ($)"] - df.iloc[-2]["Spend ($)"]
            col3.metric("Spend Change (Last Tick)", f"${spend_diff:.6f}")
        
        # Display Graph
        st.subheader("Spend Over Time")
        chart_df = df.set_index("Timestamp")
        
        # If the spend hasn't changed at all, Streamlit line_chart can look flat.
        st.line_chart(chart_df["Spend ($)"])
        
        # Display Data Table
        st.subheader("Raw Data (Newest First)")
        st.dataframe(df.tail(15).iloc[::-1], use_container_width=True)
        
    else:
        st.error(f"Failed to fetch data: {resp.status_code} - {resp.text}")
except Exception as e:
    st.error(f"Error connecting to server: {e}")

# Continuous refresh loop
time.sleep(refresh_rate)
st.rerun()
