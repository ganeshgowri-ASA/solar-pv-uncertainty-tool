# Fresh Deployment Guide: pv-uncertainty-analysis-tool

This guide walks you through creating a brand new GitHub repository and deploying it fresh to Railway and Streamlit Cloud, avoiding any issues with the previous deployment.

## Overview

- **New Repository Name:** `pv-uncertainty-analysis-tool`
- **Railway:** New PostgreSQL database with fresh credentials
- **Streamlit Cloud:** New deployment with correct DATABASE_URL from start
- **Admin Password:** `PV-Admin-2024-Secure!`

---

## Step 1: Create New GitHub Repository

### Option A: Via GitHub Web Interface

1. Go to [GitHub](https://github.com) and log in
2. Click the **+** button in the top right, select **New repository**
3. Enter repository details:
   - **Repository name:** `pv-uncertainty-analysis-tool`
   - **Description:** `Professional PV measurement uncertainty analysis tool with ISO 17025 compliance`
   - **Visibility:** Public or Private (your choice)
   - **DO NOT** initialize with README (we'll push existing code)
4. Click **Create repository**
5. Copy the repository URL (e.g., `https://github.com/YOUR_USERNAME/pv-uncertainty-analysis-tool.git`)

### Option B: Via GitHub CLI

```bash
# Install gh CLI if not already installed
# brew install gh  (macOS)
# or download from https://cli.github.com/

# Authenticate
gh auth login

# Create repository
gh repo create pv-uncertainty-analysis-tool --public --description "Professional PV measurement uncertainty analysis tool with ISO 17025 compliance"
```

---

## Step 2: Prepare and Push Code

### 2.1 Create Local Copy

```bash
# Create new directory for the fresh deployment
mkdir ~/pv-uncertainty-analysis-tool
cd ~/pv-uncertainty-analysis-tool

# Initialize git
git init

# Copy all application files from existing repository
# (Run this from your current solar-pv-uncertainty-tool directory)
```

### 2.2 Files to Copy

Copy these files and directories:

```
# Core application files
streamlit_app.py
requirements.txt

# Pages
pages/9_🔧_Admin.py

# Database module
database/__init__.py
database/connection.py
database/models.py
database/seed_data.py
database/streamlit_integration.py
database/SCHEMA.md

# Migrations
migrations/001_initial_schema_UP.sql
migrations/001_initial_schema_DOWN.sql
migrations/README.md

# Supporting modules
config_data.py
data_handler.py
uncertainty_calculator.py
financial_impact.py
file_utilities.py
pv_uncertainty_enhanced.py
monte_carlo.py
monte_carlo_analysis.py
report_generator.py
visualizations.py
standards_compliance.py
uncertainty_components.py
bifacial_uncertainty.py
test_scenarios.py

# Configuration
.gitignore
LICENSE
README.md
```

### 2.3 Quick Copy Script

Run this script from the `solar-pv-uncertainty-tool` directory:

```bash
#!/bin/bash
# copy_to_new_repo.sh

# Set destination
DEST_DIR="$HOME/pv-uncertainty-analysis-tool"

# Create directory structure
mkdir -p "$DEST_DIR/pages"
mkdir -p "$DEST_DIR/database"
mkdir -p "$DEST_DIR/migrations"
mkdir -p "$DEST_DIR/.streamlit"

# Copy main application files
cp streamlit_app.py "$DEST_DIR/"
cp requirements.txt "$DEST_DIR/"

# Copy pages
cp "pages/9_🔧_Admin.py" "$DEST_DIR/pages/"

# Copy database module
cp database/__init__.py "$DEST_DIR/database/"
cp database/connection.py "$DEST_DIR/database/"
cp database/models.py "$DEST_DIR/database/"
cp database/seed_data.py "$DEST_DIR/database/"
cp database/streamlit_integration.py "$DEST_DIR/database/"
cp database/SCHEMA.md "$DEST_DIR/database/"

# Copy migrations
cp migrations/*.sql "$DEST_DIR/migrations/" 2>/dev/null
cp migrations/README.md "$DEST_DIR/migrations/"

# Copy supporting modules
cp config_data.py "$DEST_DIR/"
cp data_handler.py "$DEST_DIR/" 2>/dev/null
cp uncertainty_calculator.py "$DEST_DIR/" 2>/dev/null
cp financial_impact.py "$DEST_DIR/"
cp file_utilities.py "$DEST_DIR/"
cp pv_uncertainty_enhanced.py "$DEST_DIR/"
cp monte_carlo.py "$DEST_DIR/"
cp monte_carlo_analysis.py "$DEST_DIR/"
cp report_generator.py "$DEST_DIR/"
cp visualizations.py "$DEST_DIR/"
cp standards_compliance.py "$DEST_DIR/"
cp uncertainty_components.py "$DEST_DIR/"
cp bifacial_uncertainty.py "$DEST_DIR/"
cp test_scenarios.py "$DEST_DIR/"

# Copy configuration files
cp .gitignore "$DEST_DIR/"
cp LICENSE "$DEST_DIR/"
cp README.md "$DEST_DIR/"

echo "Files copied to $DEST_DIR"
```

### 2.4 Push to GitHub

```bash
cd ~/pv-uncertainty-analysis-tool

# Add all files
git add .

# Commit
git commit -m "Initial commit: PV Uncertainty Analysis Tool v3.0"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/pv-uncertainty-analysis-tool.git

# Push
git branch -M main
git push -u origin main
```

---

## Step 3: Create Railway PostgreSQL Database

### 3.1 Create Railway Account & Project

1. Go to [Railway](https://railway.app) and sign in (use GitHub OAuth)
2. Click **New Project**
3. Select **Empty Project**
4. Name it: `pv-uncertainty-db`

### 3.2 Add PostgreSQL Database

1. In your new project, click **+ New**
2. Select **Database** → **Add PostgreSQL**
3. Wait for the database to provision (takes ~30 seconds)

### 3.3 Get Database Credentials

1. Click on the PostgreSQL service
2. Go to the **Variables** tab
3. You'll see these variables automatically created:
   - `DATABASE_URL` - **This is what you need!**
   - `PGHOST`
   - `PGPORT`
   - `PGUSER`
   - `PGPASSWORD`
   - `PGDATABASE`

4. **Copy the `DATABASE_URL` value** - it looks like:
   ```
   postgresql://postgres:XXXXXXXXXXXX@containers-us-west-XXX.railway.app:XXXXX/railway
   ```

### 3.4 Important Notes

- Railway PostgreSQL URLs use the format `postgresql://` which is correct for SQLAlchemy 2.0
- Railway requires SSL connections from external hosts (our code handles this automatically)
- The database is empty initially - you'll initialize it from the Streamlit Admin page

---

## Step 4: Deploy to Streamlit Cloud

### 4.1 Create Streamlit Cloud Account

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Sign in with your GitHub account

### 4.2 Deploy New App

1. Click **New app**
2. Select your repository: `YOUR_USERNAME/pv-uncertainty-analysis-tool`
3. Set:
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Advanced settings** before deploying

### 4.3 Configure Secrets (CRITICAL!)

In the **Advanced settings** → **Secrets** section, paste this **EXACTLY**:

```toml
# Database connection (Railway PostgreSQL)
DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@YOUR_HOST.railway.app:YOUR_PORT/railway"

# Admin authentication
ADMIN_PASSWORD = "PV-Admin-2024-Secure!"
```

**Replace the DATABASE_URL with your actual Railway DATABASE_URL from Step 3.3!**

### 4.4 Example Complete Secrets

```toml
# Railway PostgreSQL Database
DATABASE_URL = "postgresql://postgres:aBcDeFgHiJkL@containers-us-west-123.railway.app:5432/railway"

# Admin password for database management
ADMIN_PASSWORD = "PV-Admin-2024-Secure!"
```

### 4.5 Deploy

1. Click **Deploy!**
2. Wait for the app to build and deploy (2-3 minutes)
3. Your app URL will be: `https://YOUR_APP_NAME.streamlit.app`

---

## Step 5: Initialize Database

Once your app is deployed:

1. Open your app URL
2. Navigate to the **Admin** page (🔧 Admin in sidebar)
3. Enter the admin password: `PV-Admin-2024-Secure!`
4. On the **Connection Status** tab:
   - Verify it shows "Connected" in green
   - Check Host shows your Railway hostname
5. On the **Initialize Database** tab:
   - Click **🚀 Initialize Database Schema**
   - Wait for confirmation that tables were created
6. On the **Health Check** tab:
   - Click **🔍 Run Health Check**
   - Verify all tables show up with 0 rows

---

## Step 6: Verify Everything Works

### 6.1 Test Database Connection

```
Admin Dashboard → Connection Status
✅ Status: Connected
✅ SSL/TLS: Enabled
✅ All 13 tables visible
```

### 6.2 Test Main Application

1. Go to the main page (Home)
2. Go through each tab:
   - Module Configuration
   - Sun Simulator
   - Reference Device
   - Measurement Data
   - Uncertainty Analysis (click Calculate)
   - Results
   - Financial Impact
   - Professional Reporting

### 6.3 Test Report Generation

1. Complete an uncertainty calculation
2. Navigate to the Report tab
3. Generate PDF and Excel reports
4. Verify downloads work

---

## Troubleshooting

### "Database not connected"

1. Check Streamlit secrets have correct DATABASE_URL
2. Verify Railway database is running
3. Check the URL format is `postgresql://` (not `postgres://`)
4. Use the Reconnect button in Admin

### "Invalid admin password"

1. Check ADMIN_PASSWORD is set in Streamlit secrets
2. Verify exact value: `PV-Admin-2024-Secure!`
3. Note: Passwords are case-sensitive

### "SSL certificate verify failed"

- Railway requires SSL for external connections
- Our code handles this with `sslmode=require`
- If error persists, check Railway database is healthy

### "Connection timeout"

- Railway databases may take a moment to wake up
- Try again after 30 seconds
- Check Railway dashboard for database status

---

## Summary Checklist

- [ ] Created new GitHub repository `pv-uncertainty-analysis-tool`
- [ ] Copied all application code
- [ ] Pushed to GitHub
- [ ] Created Railway PostgreSQL database
- [ ] Copied DATABASE_URL from Railway
- [ ] Created new Streamlit Cloud app
- [ ] Added secrets with DATABASE_URL and ADMIN_PASSWORD
- [ ] Deployed app successfully
- [ ] Initialized database from Admin page
- [ ] Verified all features work

---

## Quick Reference

| Item | Value |
|------|-------|
| Repository | `pv-uncertainty-analysis-tool` |
| Admin Password | `PV-Admin-2024-Secure!` |
| Main File | `streamlit_app.py` |
| Database | Railway PostgreSQL |

---

**Created:** December 2024
**Version:** 3.0.0
