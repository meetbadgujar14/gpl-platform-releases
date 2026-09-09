#!/bin/bash
# Starts both GPL Factory (8080) and Customer Runtime (8081)
cd "$(dirname "$0")"

if [ ! -f "venv/bin/python" ]; then
    echo "Error: Virtual environment not found. Run: python3 setup.py"
    exit 1
fi

echo "Starting GPL Factory on port 8080..."
venv/bin/python run.py &
FACTORY_PID=$!

echo "Starting GPL Customer Runtime on port 8081..."
venv/bin/python run_customer.py &
RUNTIME_PID=$!

echo ""
echo "  Factory:          http://localhost:8080"
echo "  Customer Runtime: http://localhost:8081"
echo ""
echo "Press Ctrl+C to stop both."

trap "echo 'Stopping...'; kill $FACTORY_PID $RUNTIME_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
