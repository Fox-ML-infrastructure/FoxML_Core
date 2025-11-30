#!/bin/bash
# Aurora Installation Script
# This script installs Aurora and its dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print license compliance notice
print_license_notice() {
    echo ""
    echo "======================================================================"
    echo -e "${YELLOW}                 ⚠️  LICENSE COMPLIANCE NOTICE${NC}"
    echo "======================================================================"
    echo ""
    echo "This software (Aurora) is licensed under GNU AGPL v3.0 with a"
    echo "mandatory Comprehensive Minority Protection Clause."
    echo ""
    echo "INSTITUTIONAL USE REQUIREMENTS:"
    echo "  ✓ Comprehensive nondiscrimination protections for ALL minorities must exist"
    echo "  ✓ Institutions cannot cherry-pick protections (must protect ALL groups)"
    echo "  ✓ All benefit must be reinvested into protection of marginalized groups"
    echo "  ✓ No commercial finance, trading, or profit-seeking use"
    echo ""
    echo "PROHIBITED USES:"
    echo "  ✗ Hedge funds, proprietary trading, market-making"
    echo "  ✗ Commercial profit-seeking or financial exploitation"
    echo "  ✗ Use by institutions that discriminate against ANY marginalized group"
    echo ""
    echo "See EXCEPTION.md and ENFORCEMENT.md for full terms."
    echo "======================================================================"
    echo ""
    read -p "Do you understand and agree to these license terms? (yes/no): " agree
    if [[ ! "$agree" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo -e "${RED}Installation cancelled. You must agree to the license terms to proceed.${NC}"
        exit 1
    fi
    echo ""
}

# Check if Python is installed
check_python() {
    echo -e "${BLUE}Checking Python installation...${NC}"
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
        
        # Check Python version (3.8+)
        PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
        PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
            echo -e "${RED}✗ Python 3.8 or higher is required. Found Python $PYTHON_VERSION${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
    echo ""
}

# Check if pip is installed
check_pip() {
    echo -e "${BLUE}Checking pip installation...${NC}"
    if command -v pip3 &> /dev/null || python3 -m pip --version &> /dev/null; then
        echo -e "${GREEN}✓ pip found${NC}"
    else
        echo -e "${RED}✗ pip is not installed. Installing pip...${NC}"
        python3 -m ensurepip --upgrade
    fi
    echo ""
}

# Create virtual environment (optional but recommended)
setup_venv() {
    echo -e "${BLUE}Setting up virtual environment...${NC}"
    read -p "Create a virtual environment? (recommended) (yes/no) [yes]: " create_venv
    create_venv=${create_venv:-yes}
    
    if [[ "$create_venv" =~ ^[Yy][Ee][Ss]$ ]]; then
        VENV_NAME="aurora_env"
        if [ -d "$VENV_NAME" ]; then
            echo -e "${YELLOW}Virtual environment already exists. Using existing environment.${NC}"
        else
            echo -e "${BLUE}Creating virtual environment: $VENV_NAME${NC}"
            python3 -m venv "$VENV_NAME"
            echo -e "${GREEN}✓ Virtual environment created${NC}"
        fi
        
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source "$VENV_NAME/bin/activate"
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
        echo -e "${YELLOW}Note: To activate this environment in the future, run: source $VENV_NAME/bin/activate${NC}"
    else
        echo -e "${YELLOW}Proceeding without virtual environment (not recommended)${NC}"
    fi
    echo ""
}

# Upgrade pip
upgrade_pip() {
    echo -e "${BLUE}Upgrading pip...${NC}"
    python3 -m pip install --upgrade pip setuptools wheel
    echo -e "${GREEN}✓ pip upgraded${NC}"
    echo ""
}

# Install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    echo "This may take several minutes..."
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}✗ requirements.txt not found in current directory${NC}"
        echo "Please run this script from the Aurora project root directory."
        exit 1
    fi
    
    # Install requirements
    python3 -m pip install -r requirements.txt
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
}

# Run compliance checker
run_compliance_check() {
    echo -e "${BLUE}Running license compliance check...${NC}"
    if [ -f "scripts/compliance_check.py" ]; then
        python3 scripts/compliance_check.py
    else
        echo -e "${YELLOW}⚠ Compliance checker not found (this is okay)${NC}"
    fi
    echo ""
}

# Print completion message
print_completion() {
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}                    ✓ Installation Complete!${NC}"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Review the license terms:"
    echo "   - Read EXCEPTION.md for full license terms"
    echo "   - Read ENFORCEMENT.md for enforcement policies"
    echo "   - Read VALUES.md for project values"
    echo ""
    echo "2. For institutional use:"
    echo "   - Complete ATTESTATION_INSTITUTIONAL_USE.md"
    echo "   - Ensure your institution protects ALL marginalized groups"
    echo "   - See FLOWCHART.md to verify eligibility"
    echo ""
    echo "3. Activate virtual environment (if created):"
    echo "   source aurora_env/bin/activate"
    echo ""
    echo "4. Run the compliance checker anytime:"
    echo "   python3 scripts/compliance_check.py"
    echo ""
    echo "======================================================================"
    echo ""
}

# Main installation flow
main() {
    echo ""
    echo "======================================================================"
    echo -e "${BLUE}              Aurora Installation Script${NC}"
    echo "======================================================================"
    echo ""
    
    # Show license notice first
    print_license_notice
    
    # Check prerequisites
    check_python
    check_pip
    
    # Setup virtual environment
    setup_venv
    
    # Upgrade pip
    upgrade_pip
    
    # Install dependencies
    install_dependencies
    
    # Run compliance check
    run_compliance_check
    
    # Print completion
    print_completion
}

# Run main function
main

