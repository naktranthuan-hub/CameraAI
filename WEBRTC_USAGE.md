# 📱 Hướng Dẫn Sử Dụng WebRTC - Truy Cập Camera Điện Thoại

## 🎯 Tổng Quan
File `app_tichhop.py` đã được tích hợp WebRTC để có thể nhận video từ camera điện thoại khi được deploy lên hosting platforms.

## 🚀 Chạy Locally

### Windows PowerShell:
```powershell
.\run_webrtc_app.ps1
```

### Linux/Mac:
```bash
./run_webrtc_app.sh
```

### Manual:
```bash
streamlit run app_tichhop.py --server.port 8501
```

## 📱 Sử Dụng Với Điện Thoại

### 1. Local Network
- Chạy app trên máy tính
- Truy cập từ điện thoại: `http://[IP_máy_tính]:8501`
- Ví dụ: `http://192.168.1.100:8501`

### 2. Production Hosting (HTTPS Required)
- Deploy lên Streamlit Cloud/Heroku/AWS
- WebRTC yêu cầu HTTPS trong production
- Từ điện thoại truy cập URL hosting

## 🎮 Cách Sử dụng

1. **Mở app trên điện thoại**
2. **Sidebar:** Chọn "📱 Phone Camera (WebRTC)"
3. **Click "START"** để bắt đầu streaming
4. **Allow camera permission** khi browser yêu cầu
5. **Video sẽ hiển thị** với detection khoanh vùng điện thoại
6. **Thống kê vi phạm** hiển thị bên dưới
7. **Click "STOP"** để kết thúc

## ⚙️ Tính Năng WebRTC

### 🔧 Detection Features:
- ✅ Real-time phone detection từ camera
- ✅ Khoanh vùng bounding box cho điện thoại phát hiện
- ✅ Confidence score hiển thị
- ✅ Logging vi phạm vào CSV file

### 📊 Statistics Dashboard:
- 📈 Tổng số frame đã xử lý
- 📱 Tổng số điện thoại phát hiện
- ⚠️ Vi phạm confidence > 0.7
- 📁 Export log file

### 🌐 Network Configuration:
- 🔄 STUN servers cho NAT traversal
- 📡 ICE servers configuration
- 🔒 Secure WebRTC protocols

## 📋 Requirements

### 📦 Dependencies:
```
streamlit-webrtc>=0.45.0
av>=10.0.0
pandas>=1.5.0
opencv-python==4.10.0.82
ultralytics
```

### 🔧 System Requirements:
- Python 3.9+
- HTTPS (for production)
- Modern browser với WebRTC support
- Camera permissions

## 🚀 Production Deployment

### Streamlit Cloud:
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy app
4. HTTPS tự động được cung cấp

### Heroku:
```bash
git push heroku main
```

### AWS/GCP:
- Setup HTTPS certificate
- Configure reverse proxy
- Deploy with proper ports

## 🔍 Troubleshooting

### ❌ Camera không hoạt động:
- Kiểm tra browser permissions
- Thử browser khác (Chrome recommended)
- Đảm bảo HTTPS trong production

### ❌ WebRTC connection failed:
- Kiểm tra firewall settings
- Thử network khác
- Check STUN server connectivity

### ❌ Detection không chính xác:
- Cải thiện ánh sáng
- Giữ điện thoại ổn định
- Điều chỉnh confidence threshold

## 📞 Support
- Email: your-email@domain.com
- GitHub Issues: your-repo-url/issues

---
🎉 **Enjoy real-time phone detection with WebRTC!** 📱✨