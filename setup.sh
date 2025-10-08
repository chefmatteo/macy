#!/bin/bash

# Macy ML Project - Automated Setup Script with UV Package Manager
# This script installs uv package manager and sets up all dependencies

set -e  # Exit on any error

echo "🚀 Starting Macy ML Project Setup..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on macOS or Linux
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

print_status "Detected OS: $MACHINE"

# Step 1: Install uv package manager if not already installed
print_header "1. Installing UV Package Manager"

if command -v uv &> /dev/null; then
    print_status "UV is already installed: $(uv --version)"
else
    print_status "Installing UV package manager..."
    if [ "$MACHINE" = "Mac" ]; then
        # Install via Homebrew if available, otherwise use curl
        if command -v brew &> /dev/null; then
            print_status "Installing UV via Homebrew..."
            brew install uv
        else
            print_status "Installing UV via curl..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            # Add uv to PATH for current session
            export PATH="$HOME/.cargo/bin:$PATH"
        fi
    else
        # Linux installation
        print_status "Installing UV via curl..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Verify installation
    if command -v uv &> /dev/null; then
        print_status "UV successfully installed: $(uv --version)"
    else
        print_error "Failed to install UV. Please install manually from https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

# Step 2: Check Python version
print_header "2. Checking Python Version"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    print_status "Python version check passed: Python $PYTHON_VERSION"
else
    print_error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
    print_status "Please install Python 3.9+ and try again."
    exit 1
fi

# Step 3: Create virtual environment and install dependencies
print_header "3. Setting up Virtual Environment and Dependencies"

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
print_status "Working in directory: $PROJECT_DIR"

# Create virtual environment with uv
print_status "Creating virtual environment with uv..."
uv venv

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from pyproject.toml
print_status "Installing project dependencies..."
uv pip install -e .

# Install development dependencies
print_status "Installing development dependencies..."
uv pip install -e ".[dev]"

# Step 4: Verify installation
print_header "4. Verifying Installation"

print_status "Checking installed packages..."

# Check critical packages
CRITICAL_PACKAGES=("torch" "transformers" "pandas" "numpy" "sklearn" "openpyxl")

for package in "${CRITICAL_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        VERSION=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        print_status "✓ $package ($VERSION)"
    else
        print_error "✗ $package - Failed to import"
        exit 1
    fi
done

# Step 5: Download BERT model (optional, for faster first run)
print_header "5. Pre-downloading BERT Model (Optional)"

read -p "Do you want to pre-download the BERT model? This will speed up the first run. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Pre-downloading BERT model: indiejoseph/bert-base-cantonese"
    python3 -c "
from transformers import BertTokenizer, BertModel
print('Downloading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('indiejoseph/bert-base-cantonese')
print('Downloading BERT model...')
model = BertModel.from_pretrained('indiejoseph/bert-base-cantonese')
print('BERT model downloaded successfully!')
" || print_warning "Failed to download BERT model. It will be downloaded on first use."
else
    print_status "Skipping BERT model pre-download."
fi

# Step 6: Create convenience scripts
print_header "6. Creating Convenience Scripts"

# Create run script for easy execution
cat > run_models.sh << 'EOF'
#!/bin/bash
# Convenience script to run ML models

set -e

# Activate virtual environment
source .venv/bin/activate

echo "🤖 Macy ML Project - Model Runner"
echo "================================"

echo "Available models:"
echo "1. BERT + Random Forest (BERT_RF.py)"
echo "2. BERT + Logistic Regression (BERT_LR/BERT_LR.py)"
echo "3. BERT + SVM (BERT_SVM.py/BERT_SVM.py)"
echo "4. New Assignment (new_assignment.py)"

read -p "Select model to run (1-4): " choice

case $choice in
    1)
        echo "Running BERT + Random Forest..."
        cd BERT_RF.py && python3 BERT_RF.py
        ;;
    2)
        echo "Running BERT + Logistic Regression..."
        cd BERT_LR && python3 BERT_LR.py
        ;;
    3)
        echo "Running BERT + SVM..."
        cd BERT_SVM.py && python3 BERT_SVM.py
        ;;
    4)
        echo "Running New Assignment..."
        python3 new_assignment.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
EOF

chmod +x run_models.sh

# Create activate script
cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the project virtual environment

source .venv/bin/activate
echo "🐍 Virtual environment activated!"
echo "You can now run Python scripts with all dependencies available."
echo "To run models, use: ./run_models.sh"
EOF

chmod +x activate.sh

print_status "Created convenience scripts:"
print_status "  - run_models.sh: Interactive model runner"
print_status "  - activate.sh: Activate virtual environment"

# Step 7: Final instructions
print_header "7. Setup Complete! 🎉"

echo ""
echo "=================================="
echo "🎉 Setup completed successfully!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. To activate the virtual environment:"
echo "   source .venv/bin/activate"
echo "   # OR"
echo "   ./activate.sh"
echo ""
echo "2. To run models interactively:"
echo "   ./run_models.sh"
echo ""
echo "3. To run individual models:"
echo "   cd BERT_RF.py && python3 BERT_RF.py"
echo "   cd BERT_LR && python3 BERT_LR.py"
echo "   cd BERT_SVM.py && python3 BERT_SVM.py"
echo "   python3 new_assignment.py"
echo ""
echo "4. To install additional packages:"
echo "   uv pip install <package-name>"
echo ""
echo "📁 Project structure:"
echo "   - Virtual environment: .venv/"
echo "   - Dependencies: pyproject.toml"
echo "   - Models: BERT_RF.py/, BERT_LR/, BERT_SVM.py/"
echo "   - Data: *.xlsx, *.csv files"
echo ""

# Check if we're in an activated environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment is currently active!"
else
    print_warning "Remember to activate the virtual environment before running scripts:"
    echo "   source .venv/bin/activate"
fi

echo ""
print_status "Happy coding! 🚀"
