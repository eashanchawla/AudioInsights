#!/bin/bash

# Exit on error
set -e

echo "Starting AudioInsights setup..."

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it and try again."
    echo "On Mac: brew install ffmpeg"
    echo "On Linux: sudo apt install ffmpeg"
    exit 1
fi

# Check for Python 3.10+
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "Error: Python 3.10+ is required. Found version $PYTHON_VERSION"
    exit 1
fi

echo "Python $PYTHON_VERSION detected."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools

# Install dependencies
echo "Installing dependencies from requirements.txt..."
# We use || true because some environments might have trouble with specific whisper versions
# but we want to continue with the rest of the setup if possible.
pip install -r requirements.txt || echo "Warning: Some dependencies failed to install. You may need to install them manually."

# Mac Apple Silicon specific installation
if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    echo "Apple Silicon detected. Installing mlx-whisper..."
    pip install mlx-whisper
fi

# Setup .env file
if [ ! -f ".env" ]; then
    echo "Setting up .env file..."
    cp .env.example .env

    printf "Enter your OpenAI API key (or press Enter to skip): "
    read -rs api_key
    echo "" # Add newline after silent read

    if [ ! -z "$api_key" ]; then
        # Use Python to handle replacement safely across platforms
        python3 -c "
import sys
filepath = '.env'
with open(filepath, 'r') as f:
    content = f.read()
new_content = content.replace('your_api_key_here', sys.argv[1])
with open(filepath, 'w') as f:
    f.write(new_content)
" "$api_key"
        echo ".env file updated with API key."
    else
        echo "No API key entered. You will need to manually add it to .env before running the application."
    fi
else
    echo ".env file already exists. Skipping .env setup."
fi

echo ""
echo "Setup complete!"
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the demo:"
echo "streamlit run demo.py"
echo ""
echo "To run the full application:"
echo "streamlit run app.py"
