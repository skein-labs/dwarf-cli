#!/bin/bash
set -e

INSTALL_DIR="${HOME}/.local/bin"
MODEL_DIR="${HOME}/.dwarf/model"

echo "=== Dwarf CLI Installer ==="
echo ""

# 1. Build
echo "[1/3] Building dwarf (release)..."
cargo build --release

# 2. Install binary
echo "[2/3] Installing binary to $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
cp target/release/dwarf "$INSTALL_DIR/dwarf"
chmod +x "$INSTALL_DIR/dwarf"

# 3. Shell config
echo "[3/3] Configuring shell..."

SHELL_RC=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [ -n "$SHELL_RC" ]; then
    # Add ~/.local/bin to PATH if not already there
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_RC" 2>/dev/null; then
        echo '' >> "$SHELL_RC"
        echo '# Dwarf CLI' >> "$SHELL_RC"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    fi

    # Add DWARF_MODEL_DIR if not already there
    if ! grep -q 'DWARF_MODEL_DIR' "$SHELL_RC" 2>/dev/null; then
        echo "export DWARF_MODEL_DIR=\"$MODEL_DIR\"" >> "$SHELL_RC"
    fi

    echo "  Added to $SHELL_RC"
else
    echo "  No .bashrc or .zshrc found. Add manually:"
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    echo "    export DWARF_MODEL_DIR=\"$MODEL_DIR\""
fi

echo ""
echo "Done! Run 'source $SHELL_RC' or open a new terminal, then:"
echo ""
echo "  dwarf \"list files sorted by size\""
echo "  dwarf -x \"count lines in all python files\""
echo "  dwarf -t   # TUI mode"
echo ""

# Check if model is downloaded
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Model not found. Run ./setup.sh first to download it."
fi
