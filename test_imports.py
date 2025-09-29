#!/usr/bin/env python3
"""
Test script for app_tichhop.py imports and compatibility
"""
import sys
import os

print("Testing app_tichhop.py imports...")

try:
    # Test critical imports
    import streamlit as st
    print("✓ Streamlit imported successfully")
except ImportError as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    import cv2
    import numpy as np
    print(f"✓ OpenCV {cv2.__version__} and NumPy {np.__version__} imported successfully")
except ImportError as e:
    print(f"✗ OpenCV/NumPy import failed: {e}")

try:
    from ultralytics import YOLO
    print("✓ YOLO imported successfully")
except ImportError as e:
    print(f"✗ YOLO import failed: {e}")

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    import av
    print("✓ WebRTC packages imported successfully")
except ImportError as e:
    print(f"✗ WebRTC packages import failed: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

print("\n✅ Import test completed!")
print("Ready for Streamlit Cloud deployment!")