# dwarf

A tiny shell assistant powered by [Dwarf-15M](https://huggingface.co/ThingAI/Dwarf-15M) — a 15.54M parameter language model specialized in shell/bash commands.

Built in Rust with [candle](https://github.com/huggingface/candle) for inference and [ratatui](https://github.com/ratatui/ratatui) for the terminal UI.

## Features

- **Natural language → shell commands** — describe what you need, get a ready-to-run command
- **TUI mode** — interactive terminal interface with vim-like keybindings
- **One-shot mode** — pipe a question, get an answer
- **Command execution** — run generated commands directly from the TUI
- **Script checker** — lint shell scripts with `bash -n` and `shellcheck` via `/check`
- **Fully offline** — runs locally on CPU, no API keys needed

## Install

### Quick install

```bash
git clone https://github.com/skein-labs/dwarf-cli.git
cd dwarf-cli
./setup.sh      # downloads the model from HuggingFace
./install.sh    # builds and installs dwarf to ~/.local/bin
```

Then open a new terminal or `source ~/.bashrc` and you're ready to go.

### Requirements

- Rust 1.70+
- ~60MB disk space for model weights

### Manual install

If you prefer to do it yourself:

```bash
cargo build --release
cp target/release/dwarf ~/.local/bin/
```

The model files (`model.safetensors`, `config.json`, `tokenizer.json`) are loaded from the first path found:

1. `./model/` — relative to where you run `dwarf`
2. `$DWARF_MODEL_DIR` — custom path via environment variable
3. `~/.dwarf/model/` — default location (used by `setup.sh`)

## Usage

### One-shot mode

```bash
dwarf "list files sorted by size"
# Output: ls -lS

dwarf -x "count lines in all python files"
# Generates and executes the command
```

### TUI mode

```bash
dwarf -t
```

#### Keybindings

| Key | Action |
|-----|--------|
| `i` | Enter edit mode |
| `Esc` | Back to normal mode |
| `Enter` | Send prompt |
| `e` | Execute last generated command |
| `q` | Quit |
| `c` | Clear messages |
| `↑/↓` | Scroll |
| `PgUp/PgDn` | Scroll fast |
| `Home/End` | Top / bottom |
| Mouse wheel | Scroll |

#### Slash commands

| Command | Description |
|---------|-------------|
| `/check <file>` | Lint a shell script |
| `/clear` | Clear chat history |
| `/help` | Show available commands |

## Architecture

Dwarf-15M is a decoder-only transformer with:

- 15.54M parameters
- Grouped-Query Attention (GQA)
- SwiGLU feed-forward layers
- Rotary Position Embeddings (RoPE)
- RMSNorm

Inference runs on CPU via candle — no GPU required.

## License

Apache-2.0
