# Aurora Installation Guide

## One-Command Install

To install Aurora with a single command, use:

```bash
curl -fsSL https://raw.githubusercontent.com/Aurora-Jennifer/Aurora-v2/main/install.sh | bash
```

### What the Install Script Does

1. **License Compliance Check**
   - Displays the comprehensive minority protection license terms
   - Requires explicit agreement before proceeding
   - Ensures users understand the license requirements

2. **Prerequisites Check**
   - Verifies Python 3.8+ is installed
   - Checks for pip installation
   - Installs pip if missing

3. **Virtual Environment Setup**
   - Creates a virtual environment (`aurora_env`) to isolate dependencies
   - Activates the environment automatically
   - Provides instructions for future activation

4. **Dependency Installation**
   - Upgrades pip, setuptools, and wheel
   - Installs all dependencies from `requirements.txt`
   - Handles errors gracefully

5. **Compliance Verification**
   - Runs the license compliance checker
   - Displays compliance reminders

### Alternative: Download and Run Locally

If you prefer to review the script before running it:

```bash
# Download the install script
curl -fsSL https://raw.githubusercontent.com/Aurora-Jennifer/Aurora-v2/main/install.sh -o install.sh

# Review the script (recommended)
cat install.sh

# Make it executable and run
chmod +x install.sh
./install.sh
```

### Manual Installation

If you prefer to install manually:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aurora-Jennifer/Aurora-v2.git
   cd Aurora-v2
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv aurora_env
   source aurora_env/bin/activate  # On Windows: aurora_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Run compliance check**
   ```bash
   python3 scripts/compliance_check.py
   ```

### Post-Installation

After installation:

1. **Review License Terms**
   - Read `EXCEPTION.md` for full license terms
   - Read `ENFORCEMENT.md` for enforcement policies
   - Read `VALUES.md` for project values

2. **For Institutional Use**
   - Complete `ATTESTATION_INSTITUTIONAL_USE.md`
   - Ensure your institution protects ALL marginalized groups
   - See `FLOWCHART.md` to verify eligibility

3. **Activate Virtual Environment** (if created)
   ```bash
   source aurora_env/bin/activate
   ```

### Troubleshooting

**Python not found:**
- Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)

**Permission denied:**
- Make sure the script is executable: `chmod +x install.sh`

**Virtual environment issues:**
- Ensure `python3-venv` package is installed (Linux)
- On macOS, you may need Xcode Command Line Tools

**Dependency installation fails:**
- Some packages may require system libraries (e.g., TA-Lib)
- See `requirements.txt` for specific package notes
- Check individual package documentation for system requirements

### Security Note

The install script:
- Does not require sudo/root privileges
- Only installs Python packages in the virtual environment
- Does not modify system Python or global packages
- Can be reviewed before execution

For maximum security, download and review the script before running it.

