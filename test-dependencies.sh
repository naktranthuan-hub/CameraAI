#!/bin/bash
# Test script to verify project dependencies locally before push

echo "🔍 Testing CameraAI Dependencies..."
echo "=================================="

# Test Python version
echo "📋 Python version:"
python --version

echo ""
echo "📦 Testing package imports..."

# Test core dependencies
python -c "
try:
    import streamlit
    print('✅ Streamlit OK')
except ImportError as e:
    print('❌ Streamlit FAILED:', e)

try:
    import cv2
    expected_version = "4.10.0"  # Runtime version that OpenCV reports
    current_version = cv2.__version__
    if current_version == expected_version:
        print(f'✅ OpenCV {current_version} matches expected version')
    else:
        print(f'⚠️  OpenCV {current_version} (expected {expected_version}, but this might still work)')
except ImportError as e:
    print('❌ OpenCV FAILED:', e)

try:
    import numpy
    print('✅ NumPy OK')
except ImportError as e:
    print('❌ NumPy FAILED:', e)

try:
    import ultralytics
    print('✅ Ultralytics OK')
except ImportError as e:
    print('❌ Ultralytics FAILED:', e)

try:
    import mediapipe
    print('✅ MediaPipe OK')
except ImportError as e:
    print('❌ MediaPipe FAILED:', e)
"

echo ""
echo "🔧 Testing syntax compilation..."

# Test syntax
python -m py_compile app_final.py && echo "✅ app_final.py compiles OK" || echo "❌ app_final.py has syntax errors"
python -m py_compile app.py && echo "✅ app.py compiles OK" || echo "❌ app.py has syntax errors"
python -m py_compile test.py && echo "✅ test.py compiles OK" || echo "❌ test.py has syntax errors"

echo ""
echo "📁 Checking required files..."

# Check required files
[ -f "yolo11n.pt" ] && echo "✅ yolo11n.pt found" || echo "❌ yolo11n.pt missing"
[ -f "lythuongkiet.jpg" ] && echo "✅ lythuongkiet.jpg found" || echo "❌ lythuongkiet.jpg missing"
[ -f "nguyenanninh.jpg" ] && echo "✅ nguyenanninh.jpg found" || echo "❌ nguyenanninh.jpg missing"
[ -f "requirements.txt" ] && echo "✅ requirements.txt found" || echo "❌ requirements.txt missing"

echo ""
echo "🎯 Test OpenCV basic functionality..."

python -c "
import cv2
import numpy as np
try:
    # Test basic operations
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('✅ OpenCV basic operations work')
except Exception as e:
    print('❌ OpenCV operations failed:', e)
"

echo ""
echo "🚀 Test complete! If all checks passed, you're ready to push to GitHub."