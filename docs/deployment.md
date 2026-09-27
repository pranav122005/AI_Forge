# AIForge Production Deployment Guide

This guide covers deployment strategies for AIForge:
1. **Frontend**: Vercel (Static SPA)
2. **Backend**: Linux VPS / Cloud VM (Docker or Systemd + Uvicorn)
3. **Full Stack**: Docker Compose

---

## 1. Architecture Topology

```
                  ┌──────────────────────┐
                  │    Internet Users    │
                  └──────────┬───────────┘
                             │ HTTPS
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│     Vercel Frontend     │             │     VPS Backend Host    │
│   (React 18 / Vite)     │             │  (FastAPI / Port 8000)  │
│                         │             │                         │
│ VITE_API_BASE_URL ──────┼────────────►│  ALLOWED_ORIGINS        │
│                         │ REST API    │  GEMINI_API_KEY         │
└─────────────────────────┘             └─────────────────────────┘
```

---

## 2. Vercel Frontend Deployment

1. Connect your GitHub repository to Vercel.
2. Configure project settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
3. Set Environment Variables in Vercel Dashboard:
   - `VITE_API_BASE_URL`: `https://api.yourdomain.com` (or your VPS backend URL)
4. Deploy.

> **Security Warning**: NEVER set `GEMINI_API_KEY`, `OPENAI_API_KEY`, or any secret keys on Vercel. All `VITE_*` variables are public.

---

## 3. VPS Backend Deployment (Docker)

### Prerequisites
- Ubuntu 22.04 / 24.04 LTS or Debian 12 VPS
- Docker & Docker Compose installed
- Domain with DNS A record pointing to VPS IP

### Steps
1. Clone repository:
   ```bash
   git clone https://github.com/your-username/aiforge.git /opt/aiforge
   cd /opt/aiforge
   ```
2. Create production `.env`:
   ```bash
   cp .env.example .env
   nano .env
   ```
   Configure:
   ```bash
   GEMINI_API_KEY=your_real_key_here
   ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:5173
   HOST=0.0.0.0
   PORT=8000
   ```
3. Start backend service:
   ```bash
   docker compose up -d backend
   ```
4. Verify health:
   ```bash
   curl http://localhost:8000/api/health
   ```

---

## 4. Nginx Reverse Proxy with SSL (Certbot)

To expose the backend over HTTPS:

```nginx
server {
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable SSL:
```bash
sudo certbot --nginx -d api.yourdomain.com
```

---

## 5. Security & Maintenance Checklist

- [ ] Firewalls configured (`ufw allow 80,443/tcp`)
- [ ] Backend runs with non-root privileges inside Docker
- [ ] Strict CORS `ALLOWED_ORIGINS` enforced
- [ ] Automated snapshot cleanups configured
