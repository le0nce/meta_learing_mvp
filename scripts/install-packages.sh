#!/bin/bash

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

#!/bin/bash

# Assumes venv is already activated via setup.sh
echo "Installing dependencies with Poetry"
poetry install --no-root