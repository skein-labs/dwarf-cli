#!/bin/bash
set -e

MODEL_DIR="${DWARF_MODEL_DIR:-$HOME/.dwarf/model}"
echo "Downloading Dwarf-15M to $MODEL_DIR ..."
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

for f in model.safetensors config.json tokenizer.json; do
    echo "  → $f"
    curl -LO "https://huggingface.co/ThingAI/Dwarf-15M/resolve/main/$f"
done

echo "Done! Model ready at $MODEL_DIR"
