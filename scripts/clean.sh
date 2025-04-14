#!/bin/bash
ROOT_DIR="$(dirname "$(dirname "$0")")" # the warp-ik directory
OUTPUT_DIR="$ROOT_DIR/output"
sudo rm -rf "$OUTPUT_DIR"/*
echo "🧹 🧼 Cleaned output directory at: $OUTPUT_DIR"