# Test WebRTC app on Windows
Write-Host "🚀 Testing CameraAI with WebRTC support..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Check if packages are installed
Write-Host "`n📦 Checking required packages..." -ForegroundColor Yellow

$packages = @(
    'streamlit',
    'streamlit_webrtc', 
    'cv2',
    'ultralytics',
    'numpy',
    'mediapipe',
    'av',
    'pandas'
)

$missing = @()
foreach ($pkg in $packages) {
    try {
        python -c "import $pkg" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $pkg" -ForegroundColor Green
        } else {
            $missing += $pkg
            Write-Host "❌ $pkg" -ForegroundColor Red
        }
    } catch {
        $missing += $pkg
        Write-Host "❌ $pkg" -ForegroundColor Red
    }
}

if ($missing.Count -gt 0) {
    Write-Host "`n⚠️  Missing packages: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Run: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "`n🎉 All packages OK! Ready to run WebRTC app." -ForegroundColor Green
}

Write-Host "`n🌐 Starting Streamlit app with WebRTC support..." -ForegroundColor Cyan
Write-Host "📱 Access from phone: http://localhost:8501" -ForegroundColor Yellow
Write-Host "`n💡 For phone access over network:" -ForegroundColor Yellow
Write-Host "   - Use your computer's IP instead of localhost" -ForegroundColor Gray
Write-Host "   - For hosting: Use HTTPS (required for WebRTC)" -ForegroundColor Gray

Write-Host "`n🚀 Launching app..." -ForegroundColor Magenta
streamlit run app_tichhop.py --server.port 8501 --server.address 0.0.0.0