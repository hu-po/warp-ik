#!/bin/bash
ROOT_DIR="$(dirname "$(dirname "$0")")" # the warp-ik directory
OUTPUT_DIR="$ROOT_DIR/output"
sudo rm -rf "$OUTPUT_DIR"/*
echo "🧹 🧼 Cleaned output directory at: $OUTPUT_DIR"
WANDB_LOGS_DIR="$ROOT_DIR/morphs/wandb"
sudo rm -rf "$WANDB_LOGS_DIR"
echo "🧹 🧼 Removed wandb logs directory at: $WANDB_LOGS_DIR"