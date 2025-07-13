#!/bin/bash

# Activate the virtual environment
if [ -d ".venv" ]; then
    echo "Activating venv"
    if [ -f "./.venv/Scripts/activate" ]; then
        source ./.venv/Scripts/activate
    elif [ -f "./.venv/bin/activate" ]; then
        source ./.venv/bin/activate
    fi
else
    echo "Virtual environment not found"
    exit 1
fi

# Ensure required formatters are installed
missing=0
if ! poetry run black --version &> /dev/null; then
    echo "black not found. Installing via Poetry..."
    poetry add --dev black || missing=1
fi
if ! poetry run isort --version &> /dev/null; then
    echo "isort not found. Installing via Poetry..."
    poetry add --dev isort || missing=1
fi
if [ "$missing" -ne 0 ]; then
    echo "Failed to install one or more formatters."
    exit 1
fi

# Run format or format check
if [ "$1" == "--check" ]; then
    echo "Checking code format"
    poetry run black --check app tests
    poetry run isort --check-only .
else
    echo "Formatting code"
    poetry run isort .
    poetry run black app tests
fi