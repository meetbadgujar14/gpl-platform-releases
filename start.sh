#!/bin/bash
echo "Starting GPL Agents..."
cd "$(dirname "$0")"

if [ ! -f "venv/bin/python" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run setup first: python3 setup.py"
    exit 1
fi

venv/bin/python run.py
