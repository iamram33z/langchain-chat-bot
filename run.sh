#!/bin/bash

# Clear __pycache__ folders
find . -type d -name "__pycache__" -exec rm -r {} +

# Clear .pyc files
find . -type f -name "*.pyc" -delete

# Restart Uvicorn without cache
uvicorn chatbot.app.main:app --reload