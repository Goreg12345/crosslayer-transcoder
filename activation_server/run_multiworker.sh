#!/bin/bash
# Multi-worker activation server startup script
# This script demonstrates how to run the server with multiple workers and a single data generator

set -e

echo "🚀 Starting Multi-Worker Activation Server"
echo "=========================================="

# Configuration
WORKERS=${WORKERS:-4}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down..."
    
    # Kill data generator if running
    if [ ! -z "$DATA_GENERATOR_PID" ]; then
        echo "Stopping data generator (PID: $DATA_GENERATOR_PID)"
        kill $DATA_GENERATOR_PID 2>/dev/null || true
        wait $DATA_GENERATOR_PID 2>/dev/null || true
    fi
    
    # Kill server if running
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    
    echo "✅ Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "Configuration:"
echo "  Workers: $WORKERS"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""

# Step 1: Start the standalone data generator
echo "📊 Starting standalone data generator..."
python -m activation_server.standalone_generator &
DATA_GENERATOR_PID=$!

echo "Data generator started (PID: $DATA_GENERATOR_PID)"

# Wait a bit for the data generator to initialize
echo "⏳ Waiting for data generator to initialize..."
sleep 5

# Check if shared memory info file exists
if [ ! -f "/tmp/activation_server_shm_info.json" ]; then
    echo "❌ Shared memory info file not found. Data generator may have failed."
    cleanup
    exit 1
fi

echo "✅ Data generator initialized successfully"

# Step 2: Start the multi-worker server
echo "🔧 Starting FastAPI server with $WORKERS workers..."
python -m activation_server.server_multiworker --host $HOST --port $PORT --workers $WORKERS &
SERVER_PID=$!

echo "Server started (PID: $SERVER_PID)"
echo ""
echo "🎯 Multi-worker activation server is now running!"
echo "   - Data Generator PID: $DATA_GENERATOR_PID"
echo "   - Server PID: $SERVER_PID"
echo "   - Workers: $WORKERS"  
echo "   - URL: http://$HOST:$PORT"
echo ""
echo "Press Ctrl+C to stop both processes"

# Wait for server to finish (or be interrupted)
wait $SERVER_PID

# Cleanup
cleanup 