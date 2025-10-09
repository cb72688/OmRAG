#!/bin/bash

# Quick setup script for Python proto tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up Protocol Buffer tests ==="

# Check if generate.sh exists
if [ ! -f "generate.sh" ]; then
    echo "Error: generate.sh not found"
    exit 1
fi

# Make generate.sh executable
chmod +x generate.sh

# Generate protocol buffers
echo "Generating protocol buffers..."
./generate.sh

# Check if python directory was created
if [ ! -d "python" ]; then
    echo "Error: python directory not created"
    exit 1
fi

# Check if episode_pb2.py exists
if [ ! -f "python/episode_pb2.py" ]; then
    echo "Error: episode_pb2.py not generated"
    exit 1
fi

echo "✓ Protocol buffers generated successfully"

# Check if tests directory exists
if [ ! -d "tests" ]; then
    echo "Creating tests directory..."
    mkdir -p tests
fi

echo "✓ Setup complete!"
echo ""
echo "You can now run tests with:"
echo "  pytest tests/test_episode_proto.py -v"
