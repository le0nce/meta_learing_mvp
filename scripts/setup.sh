#!/bin/bash

# Create virtual environment if it doesn't exist
if [ -d ".venv" ]; then
    echo "venv already installed"
else
    echo "Installing venv"
    python -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Cannot create venv"
        exit 1
    fi
fi

# Activate the virtual environment
echo "Activating venv"
if [ -f "./.venv/Scripts/activate" ]; then
    source ./.venv/Scripts/activate
elif [ -f "./.venv/bin/activate" ]; then
    source ./.venv/bin/activate
else
    echo "Python sources cannot be activated"
    exit 1
fi

# Install Poetry inside the virtual environment
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry via pip"
    pip install poetry
fi

# Enable in-project virtualenvs (creates .venv in project folder)
poetry config virtualenvs.in-project true

# Install project dependencies
echo "Installing dependencies with Poetry"
poetry install --no-root