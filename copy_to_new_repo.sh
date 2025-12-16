#!/bin/bash
# ============================================================================
# Copy script for pv-uncertainty-analysis-tool
# Run this from the solar-pv-uncertainty-tool directory
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Destination directory
DEST_DIR="${1:-$HOME/pv-uncertainty-analysis-tool}"

echo -e "${GREEN}PV Uncertainty Analysis Tool - Repository Setup${NC}"
echo "================================================"
echo ""
echo "Destination: $DEST_DIR"
echo ""

# Check if destination exists
if [ -d "$DEST_DIR" ]; then
    echo -e "${YELLOW}Warning: Destination directory already exists.${NC}"
    read -p "Delete and recreate? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf "$DEST_DIR"
    else
        echo "Aborting."
        exit 1
    fi
fi

# Create directory structure
echo -e "\n${GREEN}Creating directory structure...${NC}"
mkdir -p "$DEST_DIR/pages"
mkdir -p "$DEST_DIR/database"
mkdir -p "$DEST_DIR/migrations"
mkdir -p "$DEST_DIR/.streamlit"

# Function to copy file with error handling
copy_file() {
    local src="$1"
    local dest="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dest"
        echo -e "  ${GREEN}✓${NC} $src"
    else
        echo -e "  ${YELLOW}⚠${NC} $src (not found, skipping)"
    fi
}

# Copy main application files
echo -e "\n${GREEN}Copying main application files...${NC}"
copy_file "streamlit_app.py" "$DEST_DIR/"
copy_file "requirements.txt" "$DEST_DIR/"

# Copy pages
echo -e "\n${GREEN}Copying pages...${NC}"
copy_file "pages/9_🔧_Admin.py" "$DEST_DIR/pages/"

# Copy database module
echo -e "\n${GREEN}Copying database module...${NC}"
copy_file "database/__init__.py" "$DEST_DIR/database/"
copy_file "database/connection.py" "$DEST_DIR/database/"
copy_file "database/models.py" "$DEST_DIR/database/"
copy_file "database/seed_data.py" "$DEST_DIR/database/"
copy_file "database/streamlit_integration.py" "$DEST_DIR/database/"
copy_file "database/SCHEMA.md" "$DEST_DIR/database/"

# Copy migrations
echo -e "\n${GREEN}Copying migrations...${NC}"
copy_file "migrations/001_initial_schema_UP.sql" "$DEST_DIR/migrations/"
copy_file "migrations/001_initial_schema_DOWN.sql" "$DEST_DIR/migrations/"
copy_file "migrations/README.md" "$DEST_DIR/migrations/"

# Copy supporting modules
echo -e "\n${GREEN}Copying supporting modules...${NC}"
copy_file "config_data.py" "$DEST_DIR/"
copy_file "data_handler.py" "$DEST_DIR/"
copy_file "uncertainty_calculator.py" "$DEST_DIR/"
copy_file "financial_impact.py" "$DEST_DIR/"
copy_file "file_utilities.py" "$DEST_DIR/"
copy_file "pv_uncertainty_enhanced.py" "$DEST_DIR/"
copy_file "monte_carlo.py" "$DEST_DIR/"
copy_file "monte_carlo_analysis.py" "$DEST_DIR/"
copy_file "report_generator.py" "$DEST_DIR/"
copy_file "visualizations.py" "$DEST_DIR/"
copy_file "standards_compliance.py" "$DEST_DIR/"
copy_file "uncertainty_components.py" "$DEST_DIR/"
copy_file "bifacial_uncertainty.py" "$DEST_DIR/"
copy_file "test_scenarios.py" "$DEST_DIR/"

# Copy configuration files
echo -e "\n${GREEN}Copying configuration files...${NC}"
copy_file ".gitignore" "$DEST_DIR/"
copy_file "LICENSE" "$DEST_DIR/"
copy_file "README.md" "$DEST_DIR/"

# Create Streamlit secrets template
echo -e "\n${GREEN}Creating Streamlit secrets template...${NC}"
cat > "$DEST_DIR/.streamlit/secrets.toml.template" << 'EOF'
# =============================================================================
# Streamlit Secrets Template for pv-uncertainty-analysis-tool
# =============================================================================
#
# IMPORTANT: Copy this file to secrets.toml for local development
# For Streamlit Cloud, paste these contents in the Secrets section
#
# DO NOT commit secrets.toml to version control!
# =============================================================================

# Railway PostgreSQL Database
# Get this URL from Railway Dashboard > Your Database > Variables > DATABASE_URL
DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@YOUR_HOST.railway.app:YOUR_PORT/railway"

# Admin password for database management page
ADMIN_PASSWORD = "PV-Admin-2024-Secure!"
EOF
echo -e "  ${GREEN}✓${NC} .streamlit/secrets.toml.template"

# Create .gitignore additions
cat >> "$DEST_DIR/.gitignore" << 'EOF'

# Streamlit secrets (never commit!)
.streamlit/secrets.toml

# Local development
*.pyc
__pycache__/
.env
.venv/
venv/
*.egg-info/
EOF

# Initialize git repository
echo -e "\n${GREEN}Initializing git repository...${NC}"
cd "$DEST_DIR"
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PV Uncertainty Analysis Tool v3.0

- Complete uncertainty analysis platform for solar PV IV measurements
- ISO 17025 compliant reporting (PDF/Excel)
- GUM methodology implementation
- Financial impact analysis
- Railway PostgreSQL database integration
- Admin dashboard with database management"

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Repository setup complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Create GitHub repository:"
echo "   gh repo create pv-uncertainty-analysis-tool --public"
echo ""
echo "2. Add remote and push:"
echo "   cd $DEST_DIR"
echo "   git remote add origin https://github.com/YOUR_USERNAME/pv-uncertainty-analysis-tool.git"
echo "   git push -u origin main"
echo ""
echo "3. Create Railway PostgreSQL database at https://railway.app"
echo ""
echo "4. Deploy to Streamlit Cloud at https://share.streamlit.io"
echo "   - Add DATABASE_URL and ADMIN_PASSWORD in secrets"
echo ""
echo "See FRESH_DEPLOYMENT_GUIDE.md for detailed instructions."
