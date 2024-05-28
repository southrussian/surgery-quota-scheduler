#!/bin/bash

# Function to create a directory if it doesn't exist
create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Function to create and activate a Python virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        if [ "$(uname)" == "Darwin" ]; then
            # For macOS
            python3 -m venv venv
        else
            # For other platforms (e.g., Windows)
            python -m venv venv
        fi
    fi

    if [ "$(uname)" == "Darwin" ]; then
        # For macOS
        source venv/bin/activate
    else
        # For other platforms (e.g., Windows)
        source venv/Scripts/activate
    fi

    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
}

# Call the setup_venv function to create/activate the virtual environment and install dependencies
setup_venv

# Rest of your script
timestamp=$(date +"%d-%m-%YT%H-%M-%S")
create_directory "runs"
create_directory "runs/$timestamp"

echo "Running preprocessing.py..."
if [ "$(uname)" == "Darwin" ]; then
    # For macOS
    python3 ppo-scheduler.py "$timestamp"
else
    # For other platforms (e.g., Windows)
    python ppo-scheduler.py "$timestamp"
fi

echo "Running ppo-scheduler.py..."
if [ "$(uname)" == "Darwin" ]; then
    # For macOS
    python3 ppo-scheduler.py "$timestamp"
else
    # For other platforms (e.g., Windows)
    python ppo-scheduler.py "$timestamp"
fi

# Deactivate the virtual environment
deactivate