#!/bin/bash

# Change this line to your preferred python interpreter--we recommend python 3.10 or above
my_interpreter=python3

if ! [ -d ./src/.env ]; then
    echo "Virtual environment .env not detected. Creating virtual environment and activating..."
    eval "$my_interpreter -m venv src/.env"
fi

source src/.env/bin/activate
pip install -r requirements.txt
