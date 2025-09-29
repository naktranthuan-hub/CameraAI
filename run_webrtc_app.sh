#!/bin/bash
# Test WebRTC app locally
echo "🚀 Testing CameraAI with WebRTC support..."
echo "=========================================="

# Check if packages are installed
python -c "
import sys
required_packages = [
    'streamlit',
    'streamlit_webrtc', 
    'cv2',
    'ultralytics',
    'numpy',
    'mediapipe',
    'av',
    'pandas'
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg}')

if missing:
    print(f'\n⚠️  Missing packages: {missing}')
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('\n🎉 All packages OK! Ready to run WebRTC app.')
"

echo ""
echo "🌐 Starting Streamlit app with WebRTC support..."
echo "📱 Access from phone: https://<your-public-ip>:8501"
echo ""
echo "💡 For hosting deployment:"
echo "1. Use HTTPS (required for WebRTC)"
echo "2. Configure STUN/TURN servers if needed"
echo "3. Open necessary ports in firewall"
echo ""

streamlit run app_tichhop.py --server.port 8501 --server.address 0.0.0.0