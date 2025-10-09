#!/bin/bash

# Protocol Buffer Generation Script for Omega-RAG
# This script generates Python, Go, and other language bindings from .proto files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Protocol Buffer Code Generation ===${NC}"

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo -e "${RED}Error: protoc (Protocol Buffer compiler) is not installed${NC}"
    echo "Install it from: https://grpc.io/docs/protoc-installation/"
    exit 1
fi

echo -e "${YELLOW}Using protoc version:${NC}"
protoc --version

# Create output directories
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p python
mkdir -p go
mkdir -p cpp
mkdir -p java

# Generate Python code
echo -e "${YELLOW}Generating Python code...${NC}"
python -m grpc_tools.protoc \
    -I. \
    --python_out=python \
    --pyi_out=python \
    --grpc_python_out=python \
    episode.proto

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python code generated successfully${NC}"
else
    echo -e "${RED}✗ Python code generation failed${NC}"
    exit 1
fi

# Fix Python imports (replace absolute imports with relative)
if [ -f "python/episode_pb2_grpc.py" ]; then
    echo -e "${YELLOW}Fixing Python imports...${NC}"
    sed -i.bak 's/import episode_pb2/from . import episode_pb2/g' python/episode_pb2_grpc.py
    rm -f python/episode_pb2_grpc.py.bak
fi

# Create __init__.py for Python package
cat > python/__init__.py << 'EOF'
"""
Generated Protocol Buffer definitions for Omega-RAG Episode schema.
"""

from .episode_pb2 import (
    Episode,
    Problem,
    Plan,
    SubTask,
    Trajectory,
    ExecutionStep,
    StepStatus,
    Outcome,
    EpisodeMetadata,
    StoreEpisodeRequest,
    StoreEpisodeResponse,
    GetEpisodeRequest,
    GetEpisodeResponse,
    SearchEpisodesRequest,
    SearchEpisodesResponse,
    EpisodeSearchResult,
    EpisodeFilter,
    QueryEmbedding,
    BatchStoreEpisodesRequest,
    BatchStoreEpisodesResponse,
    UpdateEpisodeMetadataRequest,
    UpdateEpisodeMetadataResponse,
    DeleteEpisodeRequest,
    DeleteEpisodeResponse,
    EpisodeStatistics,
    GetStatisticsRequest,
    GetStatisticsResponse,
)

__all__ = [
    "Episode",
    "Problem",
    "Plan",
    "SubTask",
    "Trajectory",
    "ExecutionStep",
    "StepStatus",
    "Outcome",
    "EpisodeMetadata",
    "StoreEpisodeRequest",
    "StoreEpisodeResponse",
    "GetEpisodeRequest",
    "GetEpisodeResponse",
    "SearchEpisodesRequest",
    "SearchEpisodesResponse",
    "EpisodeSearchResult",
    "EpisodeFilter",
    "QueryEmbedding",
    "BatchStoreEpisodesRequest",
    "BatchStoreEpisodesResponse",
    "UpdateEpisodeMetadataRequest",
    "UpdateEpisodeMetadataResponse",
    "DeleteEpisodeRequest",
    "DeleteEpisodeResponse",
    "EpisodeStatistics",
    "GetStatisticsRequest",
    "GetStatisticsResponse",
]
EOF

echo -e "${GREEN}✓ Python package initialized${NC}"

# Generate Go code (optional)
if command -v protoc-gen-go &> /dev/null; then
    echo -e "${YELLOW}Generating Go code...${NC}"
    protoc \
        -I. \
        --go_out=go \
        --go_opt=paths=source_relative \
        --go-grpc_out=go \
        --go-grpc_opt=paths=source_relative \
        episode.proto
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Go code generated successfully${NC}"
    else
        echo -e "${YELLOW}⚠ Go code generation skipped or failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ protoc-gen-go not found, skipping Go generation${NC}"
fi

# Generate C++ code (optional)
echo -e "${YELLOW}Generating C++ code...${NC}"
protoc \
    -I. \
    --cpp_out=cpp \
    episode.proto

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ C++ code generated successfully${NC}"
else
    echo -e "${YELLOW}⚠ C++ code generation failed${NC}"
fi

# Generate Java code (optional)
echo -e "${YELLOW}Generating Java code...${NC}"
protoc \
    -I. \
    --java_out=java \
    episode.proto

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Java code generated successfully${NC}"
else
    echo -e "${YELLOW}⚠ Java code generation failed${NC}"
fi

echo ""
echo -e "${GREEN}=== Code Generation Complete ===${NC}"
echo -e "Generated files:"
echo -e "  Python: ${SCRIPT_DIR}/python/"
echo -e "  Go:     ${SCRIPT_DIR}/go/"
echo -e "  C++:    ${SCRIPT_DIR}/cpp/"
echo -e "  Java:   ${SCRIPT_DIR}/java/"
echo ""
echo -e "${YELLOW}To use in Python:${NC}"
echo -e "  export PYTHONPATH=\"${SCRIPT_DIR}/python:\$PYTHONPATH\""
echo -e "  from episode_pb2 import Episode"
