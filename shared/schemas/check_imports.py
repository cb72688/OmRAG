#!/usr/bin/env python3
"""
Diagnostic script to check if protocol buffer generation was successful.
"""

import sys
from pathlib import Path

print("=== Protocol Buffer Import Diagnostic ===\n")

# Check if python directory exists
proto_dir = Path(__file__).parent / "python"
print(f"1. Checking proto directory: {proto_dir}")
print(f"   Exists: {proto_dir.exists()}")

if proto_dir.exists():
    print(f"   Contents:")
    for item in proto_dir.iterdir():
        print(f"     - {item.name}")
    print()

# Add to path
sys.path.insert(0, str(proto_dir))

# Try importing
print("2. Attempting to import episode_pb2...")
try:
    import episode_pb2
    print("   ✓ Import successful!")
    print(f"   Module location: {episode_pb2.__file__}")
    print(f"   Available classes: {
          [name for name in dir(episode_pb2) if not name.startswith('_')]}")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print(f"   Python path: {sys.path}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

print("\n3. Checking for required dependencies...")
try:
    from google.protobuf import timestamp_pb2
    print("   ✓ google.protobuf available")
except ImportError:
    print("   ✗ google.protobuf not available - install with: pip install protobuf")

try:
    from google.protobuf import json_format
    print("   ✓ google.protobuf.json_format available")
except ImportError:
    print("   ✗ json_format not available")
