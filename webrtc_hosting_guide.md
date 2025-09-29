# WebRTC Hosting Configuration for CameraAI

## HTTPS Requirement
WebRTC requires HTTPS in production. Configure your web server (nginx/apache) with SSL certificate.

### Nginx Configuration Example:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Environment Variables for Production:
```bash
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
```

### STUN/TURN Servers:
For better connectivity, consider setting up your own TURN server or using services like:
- Google STUN: stun:stun.l.google.com:19302
- Twilio STUN/TURN
- AWS STUN/TURN

### Firewall Configuration:
Open the following ports:
- 443 (HTTPS)
- 8501 (Streamlit, if not proxied)
- UDP 10000-20000 (WebRTC media)

### Docker Deployment:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app_tichhop.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Security Considerations:
1. Use HTTPS only
2. Implement proper authentication if needed
3. Rate limiting for WebRTC connections
4. Monitor resource usage (CPU/memory)
5. Set up proper logging and monitoring