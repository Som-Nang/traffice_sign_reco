#!/bin/bash
if [ ! -f "model/traffic_sign_model.keras" ]; then
    echo "Model not found. Training..."
    python training.py
fi
echo "Starting web app..."
python app.py