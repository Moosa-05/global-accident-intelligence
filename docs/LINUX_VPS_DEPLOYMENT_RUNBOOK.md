# Production Deployment Runbook for Linux VPS
**Platform:** Global Accident Intelligence SaaS  
**Target OS:** Ubuntu 22.04 / 24.04 LTS (x86_64)  
**Architecture:** Docker Compose (PostgreSQL 15 + FastAPI Uvicorn + React 19 Nginx)  
**Document Version:** 1.0.0  
**Last Updated:** November 19, 2026  

---

## 1. Hardware & System Requirements

| Metric | Minimum Specification | Recommended Production |
| :--- | :--- | :--- |
| **CPU** | 2 vCPU cores | 4 vCPU cores |
| **Memory (RAM)** | 4 GB | 8 GB |
| **Storage** | 40 GB NVMe SSD | 80 GB NVMe SSD |
| **Network** | 100 Mbps, Static IPv4 | 1 Gbps, Static IPv4 + IPv6 |
| **Supported OS** | Ubuntu 22.04 / 24.04 LTS, Debian 12 | Ubuntu 24.04 LTS |

---

## 2. Server Initial Hardening

### 2.1 Create Non-Root Deployment User
Connect as `root` and create a dedicated deployer user:
```bash
adduser deploy --gecos ""
usermod -aG sudo deploy
```

### 2.2 Configure SSH Key Authentication
```bash
mkdir -p /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
```

Edit `/etc/ssh/sshd_config`:
```ini
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
X11Forwarding no
```
Reload SSH daemon:
```bash
sudo systemctl reload ssh
```

### 2.3 Configure UFW Firewall
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP Certbot/Nginx'
sudo ufw allow 443/tcp comment 'HTTPS Production'
sudo ufw enable
```

---

## 3. Install Docker Engine & Docker Compose

Run as user `deploy`:
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Grant deploy user docker permissions
sudo usermod -aG docker deploy
newgrp docker
```

---

## 4. Deploying Application Code & Environment

### 4.1 Clone Repository
```bash
sudo mkdir -p /opt/accident-intel
sudo chown -R deploy:deploy /opt/accident-intel
cd /opt/accident-intel

git clone https://github.com/your-org/accident-intel.git .
```

### 4.2 Configure Production Environment Variables
Copy template to `.env.production`:
```bash
cp .env.production.example .env.production
```

Generate a cryptographically random 32-character secret key:
```bash
openssl rand -hex 32
```

Edit `.env.production` with your favorite editor:
```bash
nano .env.production
```

Verify these critical production variables:
```ini
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=paste_generated_32_character_hex_key_here
POSTGRES_USER=accident_prod_user
POSTGRES_PASSWORD=generate_a_very_strong_password_here
POSTGRES_DB=accident_prod_db
DATABASE_URL=postgresql+psycopg2://accident_prod_user:password_here@db:5432/accident_prod_db
BACKEND_CORS_ORIGINS=https://accidentintel.com,https://www.accidentintel.com,https://api.accidentintel.com
ALLOWED_HOSTS=accidentintel.com,www.accidentintel.com,api.accidentintel.com,localhost
WEB_CONCURRENCY=4
```

---

## 5. TLS Certificate Provisioning (Let's Encrypt Certbot)

Install Certbot:
```bash
sudo apt-get install -y certbot
```

Ensure DNS records point to server IP, then issue certificates in standalone mode:
```bash
sudo certbot certonly --standalone \
  -d accidentintel.com \
  -d www.accidentintel.com \
  -d api.accidentintel.com \
  --agree-tos \
  --email security@accidentintel.com \
  --non-interactive
```

Verify certificates:
```bash
sudo ls -l /etc/letsencrypt/live/accidentintel.com/
# fullchain.pem
# privkey.pem
```

---

## 6. Build & Launch Application Stack

### 6.1 Build Containers
```bash
cd /opt/accident-intel
docker compose --env-file .env.production build
```

### 6.2 Launch Services in Background
```bash
docker compose --env-file .env.production up -d
```

### 6.3 Monitor Migration & Startup Logs
```bash
docker compose logs -f api
```
Expected output:
```
[Entrypoint] Verifying database connectivity...
[Entrypoint] Database connected successfully on attempt 1.
[Entrypoint] Running Alembic database migrations...
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Running upgrade  -> 001_initial_tables
INFO  [alembic.runtime.migration] Running upgrade 001_initial_tables -> 002_severity_and_combined_logs
INFO  [alembic.runtime.migration] Running upgrade 002_severity_and_combined_logs -> 003_injury_and_exposure_logs
INFO  [alembic.runtime.migration] Running upgrade 003_injury_and_exposure_logs -> 004_saas_multi_tenancy
INFO  [alembic.runtime.migration] Running upgrade 004_saas_multi_tenancy -> 005_tenant_prediction_isolation
[Entrypoint] Migrations completed successfully.
[Entrypoint] Starting Uvicorn with 4 worker processes...
```

### 6.4 Verify Healthchecks
```bash
docker compose ps
```
All containers should display status `Up (healthy)`.

---

## 7. Verification & Smoke Testing

### 7.1 Local Health Check
```bash
curl -I http://localhost:8000/health/ready
```
Expected: `HTTP/1.1 200 OK`

### 7.2 Public HTTPS Endpoint Test
```bash
curl -I https://accidentintel.com
curl -I https://accidentintel.com/health/ready
curl -s https://accidentintel.com/api/v1/models/ | jq .
```

---

## 8. Automated Backups & Disaster Recovery

### 8.1 Configure Backup Script Permissions
```bash
chmod +x /opt/accident-intel/scripts/backup_db.sh
mkdir -p /var/backups/accident
```

### 8.2 Add Daily Cron Job
```bash
sudo crontab -u deploy -e
```
Add the following line (daily backup at 02:00 UTC with 30-day automated retention):
```cron
0 2 * * * /opt/accident-intel/scripts/backup_db.sh /var/backups/accident >> /var/log/accident_backup.log 2>&1
```

### 8.3 Disaster Recovery Drill
To restore from backup:
```bash
/opt/accident-intel/scripts/restore_db.sh /var/backups/accident/accident_db_YYYYMMDD_HHMMSS.sql.gz
```
Enter `RESTORE` when prompted.

---

## 9. HSTS Policy Warning

> [!WARNING]
> In accordance with Phase 12 Go-Live policies, HSTS `preload` **MUST NOT** be enabled until the apex domain and all subdomains (`www`, `api`) are thoroughly confirmed to serve valid TLS certificates without errors for at least 7 days. Once verified, `preload` can be safely uncommented in `/opt/accident-intel/deployment/nginx/production.conf`.
