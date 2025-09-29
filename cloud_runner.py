#!/usr/bin/env python3
"""
Streamlit Cloud optimized runner
Handles resource limits and cloud-specific configurations
"""
import streamlit as st
import os
import sys
import warnings

# Suppress warnings in cloud environment
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Cloud environment detection
IS_CLOUD = os.environ.get("STREAMLIT_CLOUD", False) or "/mount/src" in os.getcwd()

if IS_CLOUD:
    # Optimize for cloud deployment
    os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"
    
    # Reduce memory usage
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    
    st.set_page_config(
        page_title="Camera AI - Cloud", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add cloud-specific warnings
    if "webrtc" in sys.argv:
        st.warning("🌐 Running in Cloud Mode - WebRTC may have connectivity limitations")

# Import main app
if __name__ == "__main__":
    # Import and run the main app
    exec(open("app_tichhop.py").read())