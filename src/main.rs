mod model;
mod checker;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use model::{DwarfConfig, DwarfModel};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Margin},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{self};
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use tokenizers::Tokenizer;

// ── CLI ────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "dwarf", version, about = "A tiny shell assistant powered by Dwarf-15M")]
struct Cli {
    /// Natural language request (non‑interactive)
    request: Vec<String>,

    /// Execute the generated command immediately
    #[arg(short = 'x', long)]
    execute: bool,

    /// Launch the TUI
    #[arg(short = 't', long)]
    tui: bool,
}

// ── App State ──────────────────────────────────────────────────────────

#[derive(PartialEq)]
enum InputMode { Normal, Editing }

struct App {
    dwarf: Dwarf,
    input: String,
    input_mode: InputMode,
    messages: Vec<Message>,
    list_state: ListState,
    loading: bool,
    status: String,
    splash: bool,
    // Scroll manuale della vista messaggi.
    // None = "segui il fondo" (comportamento automatico originale).
    // Some(offset) = l'utente ha scrollato manualmente, offset righe dal fondo.
    scroll_offset: Option<u16>,
}

struct Message {
    role: String,
    content: String,
    is_command: bool,
}

impl Message {
    fn user(content: String) -> Self {
        Self { role: "You".into(), content, is_command: false }
    }
    fn dwarf_response(content: String, is_cmd: bool) -> Self {
        Self { role: "Dwarf".into(), content, is_command: is_cmd }
    }
    fn system(content: String) -> Self {
        Self { role: "System".into(), content, is_command: false }
    }
}

// ── Model wrapper ──────────────────────────────────────────────────────

struct Dwarf {
    model: DwarfModel,
    tokenizer: Tokenizer,
    eos_ids: Vec<u32>,
}

impl Dwarf {
    fn load(model_dir: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_dir);
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Cannot read {config_path}"))?;
        let config: DwarfConfig = serde_json::from_str(&config_str)?;

        let weights_path = format!("{}/model.safetensors", model_dir);
        let weights = candle_core::safetensors::load(&weights_path, &Device::Cpu)?;
        let vb = candle_nn::VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
        let model = DwarfModel::load(&config, vb)?;

        let tok_path = format!("{}/tokenizer.json", model_dir);
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        Ok(Self { model, tokenizer, eos_ids: vec![2, 7] })
    }

    fn ask(&self, question: &str) -> Result<String> {
        let prompt = format!("<|user|>\n{question}\n<|end|>\n<|assistant|>\n");
        let encoding = self.tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        let prompt_len = ids.len();

        for _ in 0..150 {
            let input = Tensor::new(&ids[..], &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let next = logits.i((0, ids.len() - 1))?.argmax(0)?.to_scalar::<u32>()?;
            if self.eos_ids.contains(&next) { break; }
            ids.push(next);
        }

        let generated = &ids[prompt_len..];
        let text = self.tokenizer.decode(generated, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

        Ok(text
            .replace("<|end|>", "")
            .replace("<|assistant|>", "")
            .replace("<|user|>", "")
            .replace("</s>", "")
            .trim()
            .to_string())
    }
}

fn is_command(text: &str) -> bool {
    let first = text.split_whitespace().next().unwrap_or("");
    let starters = [
        "ls","cd","pwd","mkdir","rm","cp","mv","cat","head","tail","grep","find",
        "chmod","chown","kill","ps","top","df","du","tar","gzip","zip","echo",
        "date","whoami","uname","ssh","scp","rsync","curl","wget","pip","apt",
        "sudo","systemctl","journalctl","sed","awk","sort","uniq","cut","tr","tee",
        "xargs","wc","diff","patch","ln","file","stat","lsof","ss","nc","ping",
        "nohup","screen","crontab","watch","htop","free","uptime","history",
        "alias","export","env","which","type","man","touch","less","more",
        "cal","clear","last","who","id","groups","su","passwd","useradd",
        "userdel","mount","umount","fdisk","dd","dmesg","ip","ifconfig",
        "netstat","openssl","md5sum","sha256sum","basename","dirname",
        "readlink","seq","sleep","for","while","if","read","set","unset",
        "nslookup","pkill","killall","bg","fg","jobs",
    ];
    starters.contains(&first) || text.contains('|') || text.starts_with("./")
}

// ── App logic ──────────────────────────────────────────────────────────

impl App {
    fn new(dwarf: Dwarf) -> Self {
        let mut s = ListState::default();
        s.select(Some(0));
        Self {
            dwarf,
            input: String::new(),
            input_mode: InputMode::Normal,
            messages: vec![],
            list_state: s,
            loading: false,
            status: "Welcome!".into(),
            splash: true,
            scroll_offset: None,
        }
    }

    fn submit(&mut self) {
        let trimmed = self.input.trim().to_string();
        if trimmed.starts_with("/check") {
            self.check_script(&trimmed);
            self.input.clear();
            return;
        }
        if trimmed == "/clear" {
            self.clear_messages();
            self.input.clear();
            return;
        }
        if trimmed == "/help" {
            self.messages.push(Message::system("Commands: /check <file>, /clear, /help".into()));
            self.input.clear();
            self.list_state.select(Some(self.messages.len().saturating_sub(1)));
            self.scroll_offset = None;
            return;
        }
        let q = self.input.trim().to_string();
        if q.is_empty() { return; }
        self.messages.push(Message::user(q.clone()));
        self.input.clear();
        self.loading = true;
        let start = Instant::now();
        match self.dwarf.ask(&q) {
            Ok(ans) => {
                let cmd = is_command(&ans);
                self.messages.push(Message::dwarf_response(ans, cmd));
                self.status = format!("Generated in {:.0}ms", start.elapsed().as_millis());
            },
            Err(e) => {
                self.messages.push(Message::system(format!("Error: {e}")));
                self.status = "Error".into();
            }
        }
        self.loading = false;
        self.list_state.select(Some(self.messages.len().saturating_sub(1)));
        // Un nuovo messaggio riporta la vista in fondo, come un client di chat normale.
        self.scroll_offset = None;
    }

    fn execute_selected(&mut self) {
        let idx = self.list_state.selected().unwrap_or(0);
        let cmd = self.messages.iter()
            .skip(idx)
            .find(|m| m.role == "Dwarf" && m.is_command)
            .or_else(|| self.messages.iter().rev().find(|m| m.role == "Dwarf" && m.is_command));

        let cmd_text = match cmd {
            Some(m) => m.content.clone(),
            None => {
                self.messages.push(Message::system("No command to execute.".into()));
                return;
            }
        };

        self.messages.push(Message::system(format!("$ {cmd_text}")));
        let start = Instant::now();
        match Command::new("bash").arg("-c").arg(&cmd_text).output() {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                for line in stdout.lines() {
                    if !line.is_empty() { self.messages.push(Message::system(line.into())); }
                }
                for line in stderr.lines() {
                    if !line.is_empty() { self.messages.push(Message::system(format!("[stderr] {line}"))); }
                }
                let code = out.status.code().unwrap_or(-1);
                let dur = start.elapsed().as_millis();
                self.status = if code == 0 { format!("Done in {dur}ms") } else { format!("Exit {code} in {dur}ms") };
            },
            Err(e) => {
                self.messages.push(Message::system(format!("Execution error: {e}")));
                self.status = "Execution failed".into();
            }
        }
        self.list_state.select(Some(self.messages.len().saturating_sub(1)));
        self.scroll_offset = None;
    }

    fn clear_messages(&mut self) {
        self.messages.clear();
        self.messages.push(Message::system("Cleared.".into()));
        self.list_state.select(Some(0));
        self.scroll_offset = None;
    }

    fn check_script(&mut self, input: &str) {
        let script = input.strip_prefix("/check").unwrap_or(input).trim();
        if script.is_empty() {
            self.messages.push(Message::system("Usage: /check <script or filepath>".into()));
            return;
        }
        let results = if std::path::Path::new(script).exists() {
            checker::check_file(std::path::Path::new(script))
        } else {
            checker::check_script_string(script)
        };
        match results {
            Ok(res) => {
                for r in &res {
                    let status = if r.passed { "✓" } else { "✗" };
                    self.messages.push(Message::system(format!("{} {} — {}", status, r.tool, if r.passed { "passed" } else { "issues found" })));
                    for msg in &r.messages {
                        let loc = msg.line.map(|l| format!(":{l}")).unwrap_or_default();
                        self.messages.push(Message::system(format!("  {} {}{}: {}", msg.level.symbol(), if script.contains('/') { "script" } else { script }, loc, msg.text)));
                    }
                }
                self.status = "Check complete".into();
            },
            Err(e) => {
                self.messages.push(Message::system(format!("Check error: {e}")));
                self.status = "Check failed".into();
            }
        }
        self.list_state.select(Some(self.messages.len().saturating_sub(1)));
        self.scroll_offset = None;
    }

    /// Scrolla la vista di `delta` righe verso l'alto (delta positivo) o verso
    /// il basso (delta negativo). `max_scroll` è la distanza massima dal fondo
    /// (cioè quante righe in più ci sono rispetto all'altezza visibile),
    /// calcolata da ui() in base al contenuto corrente.
    fn scroll_up(&mut self, delta: u16, max_scroll: u16) {
        let current = self.scroll_offset.unwrap_or(0);
        let new_offset = (current + delta).min(max_scroll);
        self.scroll_offset = Some(new_offset);
    }

    fn scroll_down(&mut self, delta: u16) {
        let current = self.scroll_offset.unwrap_or(0);
        if current <= delta {
            // Tornati in fondo: si rimette in modalità "segui il fondo".
            self.scroll_offset = None;
        } else {
            self.scroll_offset = Some(current - delta);
        }
    }
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let model_dir = if Path::new("./model").exists() {
        "./model".to_string()
    } else if let Ok(dir) = std::env::var("DWARF_MODEL_DIR") {
        dir
    } else {
        format!("{}/.dwarf/model", std::env::var("HOME").unwrap_or_else(|_| ".".into()))
    };

    if !Path::new(&format!("{model_dir}/model.safetensors")).exists() {
        eprintln!("Model not found in {model_dir}");
        std::process::exit(1);
    }

    let dwarf = Dwarf::load(&model_dir)?;

    if !cli.tui && !cli.request.is_empty() {
        let req = cli.request.join(" ");
        let ans = dwarf.ask(&req)?;
        println!("{ans}");
        if cli.execute && is_command(&ans) {
            let out = Command::new("bash").arg("-c").arg(&ans).output()?;
            print!("{}", String::from_utf8_lossy(&out.stdout));
        }
        return Ok(());
    }

    // TUI mode
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(dwarf);
    let res = run_tui(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    res?;
    Ok(())
}

// Quante righe scrolla ogni "tick" della rotella del mouse.
const MOUSE_SCROLL_STEP: u16 = 3;

fn run_tui<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    loop {
        // last_max_scroll viene aggiornato da ui() ad ogni draw, riflettendo
        // l'altezza reale del contenuto rispetto all'area visibile corrente.
        let mut last_max_scroll: u16 = 0;
        terminal.draw(|f| {
            last_max_scroll = ui(f, app);
        })?;

        match event::read()? {
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press { continue; }
                match app.input_mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Char('i') => app.input_mode = InputMode::Editing,
                        KeyCode::Char('e') => app.execute_selected(),
                        KeyCode::Char('c') => app.clear_messages(),
                        KeyCode::Down => app.scroll_down(1),
                        KeyCode::Up => app.scroll_up(1, last_max_scroll),
                        KeyCode::PageDown => app.scroll_down(10),
                        KeyCode::PageUp => app.scroll_up(10, last_max_scroll),
                        KeyCode::End => app.scroll_offset = None,
                        KeyCode::Home => app.scroll_offset = Some(last_max_scroll),
                        _ => {}
                    },
                    InputMode::Editing => match key.code {
                        KeyCode::Esc => app.input_mode = InputMode::Normal,
                        KeyCode::Enter => { app.submit(); app.input_mode = InputMode::Normal; },
                        KeyCode::Backspace => { app.input.pop(); },
                        KeyCode::Char(c) => {
                            if app.splash {
                                app.splash = false;
                            }
                            app.input.push(c);
                        },
                        _ => {}
                    },
                }
            },
            Event::Mouse(mouse) => {
                match mouse.kind {
                    MouseEventKind::ScrollUp => app.scroll_up(MOUSE_SCROLL_STEP, last_max_scroll),
                    MouseEventKind::ScrollDown => app.scroll_down(MOUSE_SCROLL_STEP),
                    _ => {}
                }
            },
            _ => {}
        }
    }
}

// ── Paste this to replace the existing ui() and render_input_bar() functions ──
// Also update INPUT_HORIZONTAL_MARGIN

const INPUT_HORIZONTAL_MARGIN: u16 = 35;

/// Disegna la UI e restituisce `max_scroll`: la distanza massima (in righe)
/// che si può scrollare verso l'alto rispetto al fondo, dato il contenuto
/// attuale e l'altezza visibile attuale. Serve a run_tui per sapere dove
/// fermare lo scroll verso l'alto (Home/PageUp) senza andare oltre l'inizio.
fn ui(f: &mut ratatui::Frame, app: &App) -> u16 {
    let bg_color = Color::Rgb(00, 00, 00);

    // Sfondo
    let bg = Block::default().style(Style::default().bg(bg_color));
    f.render_widget(bg, f.area());

    let margined = f.area().inner(Margin { horizontal: 2, vertical: 0 });

    let mut max_scroll: u16 = 0;

    if app.splash {
        let splash_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(2),
                Constraint::Length(12),
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Min(2),
            ])
            .split(margined);

        let logo = vec![
            "",
            "       ░██                                                 ",
            "       ░██                                           ░████ ",
            "       ░██                                          ░██    ",
            " ░████████ ░██    ░██    ░██  ░██████   ░██░████ ░████████ ",
            "░██    ░██ ░██    ░██    ░██       ░██  ░███        ░██    ",
            "░██    ░██  ░██  ░████  ░██   ░███████  ░██         ░██    ",
            "░██   ░███   ░██░██ ░██░██   ░██   ░██  ░██         ░██    ",
            " ░█████░██    ░███   ░███     ░█████░██ ░██         ░██    ",
            "",
        ].join("\n");

        let splash = Paragraph::new(logo)
            .style(Style::default().fg(Color::Rgb(100, 100, 140)).bg(bg_color))
            .alignment(Alignment::Center);
        f.render_widget(splash, splash_layout[1]);

        let input_area = splash_layout[2].inner(Margin { horizontal: 20, vertical: 0 });
        render_input_bar(f, app, input_area, bg_color);

        let help = Line::from(vec![
            Span::styled("i", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(" edit  ", Style::default().fg(Color::Rgb(80, 80, 100))),
            Span::styled("enter", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(" send  ", Style::default().fg(Color::Rgb(80, 80, 100))),
            Span::styled("e", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(" execute  ", Style::default().fg(Color::Rgb(80, 80, 100))),
            Span::styled("q", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(" quit", Style::default().fg(Color::Rgb(80, 80, 100))),
        ]);
        f.render_widget(
            Paragraph::new(help).alignment(Alignment::Center).style(Style::default().bg(bg_color)),
            splash_layout[3],
        );
    } else {
        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(margined);

        // ── Messages ──────────────────────────────────────────────
        let mut lines: Vec<Line> = Vec::new();

        for msg in &app.messages {
            match msg.role.as_str() {
                "You" => {
                    // User: blue accent bar on left
                    lines.push(Line::from(""));
                    lines.push(Line::from(vec![
                        Span::styled("  › ", Style::default().fg(Color::Rgb(100, 149, 237))),
                        Span::styled(&msg.content, Style::default().fg(Color::White)),
                    ]));
                }
                "Dwarf" => {
                    if msg.is_command {
                        // Command: green arrow, bright text
                        lines.push(Line::from(vec![
                            Span::styled("  → ", Style::default().fg(Color::Rgb(80, 200, 120))),
                            Span::styled(
                                &msg.content,
                                Style::default()
                                    .fg(Color::Rgb(200, 230, 200))
                                    .add_modifier(Modifier::BOLD),
                            ),
                        ]));
                    } else {
                        // Explanation: dimmer, wrapped
                        for (i, line) in msg.content.lines().enumerate() {
                            if i == 0 {
                                lines.push(Line::from(vec![
                                    Span::styled("    ", Style::default()),
                                    Span::styled(line, Style::default().fg(Color::Rgb(180, 180, 200))),
                                ]));
                            } else {
                                lines.push(Line::from(vec![
                                    Span::styled("    ", Style::default()),
                                    Span::styled(line, Style::default().fg(Color::Rgb(180, 180, 200))),
                                ]));
                            }
                        }
                    }
                }
                _ => {
                    // System: dim, prefixed
                    if msg.content.starts_with("$ ") {
                        // Executed command
                        lines.push(Line::from(""));
                        lines.push(Line::from(vec![
                            Span::styled("  $ ", Style::default().fg(Color::Rgb(200, 150, 50))),
                            Span::styled(
                                &msg.content[2..],
                                Style::default().fg(Color::Rgb(200, 150, 50)),
                            ),
                        ]));
                    } else if msg.content.starts_with("[stderr]") {
                        lines.push(Line::from(vec![
                            Span::styled("    ", Style::default()),
                            Span::styled(&msg.content, Style::default().fg(Color::Rgb(200, 80, 80))),
                        ]));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled("    ", Style::default()),
                            Span::styled(&msg.content, Style::default().fg(Color::Rgb(120, 120, 140))),
                        ]));
                    }
                }
            }
        }

        // Add empty line at end for spacing
        lines.push(Line::from(""));

        // max_scroll = quante righe in più ci sono oltre l'altezza visibile.
        // Se il contenuto è più corto dell'area visibile, non si scrolla affatto.
        let visible_height = main_layout[0].height as usize;
        max_scroll = if lines.len() > visible_height {
            (lines.len() - visible_height) as u16
        } else {
            0
        };

        // scroll_offset: None -> segui il fondo (comportamento originale).
        // Some(offset) -> offset righe sopra il fondo, ma mai oltre max_scroll
        // (nel caso, ad es., un /clear abbia ridotto il contenuto nel frattempo).
        //
        // scroll_from_top è il parametro che Paragraph::scroll si aspetta:
        // "quante righe saltare dall'INIZIO del testo". Quando siamo in fondo
        // (None o offset=0), va mostrato tutto fino al fondo, quindi
        // scroll_from_top = max_scroll. Scrollando verso l'alto (offset > 0),
        // scroll_from_top diminuisce.
        let scroll_from_top = match app.scroll_offset {
            None => max_scroll,
            Some(offset) => max_scroll.saturating_sub(offset.min(max_scroll)),
        };

        let messages = Paragraph::new(Text::from(lines))
            .style(Style::default().bg(bg_color))
            .scroll((scroll_from_top, 0));
        f.render_widget(messages, main_layout[0]);

        // ── Input bar ─────────────────────────────────────────────
        render_input_bar(f, app, main_layout[1], bg_color);

        // ── Status bar ────────────────────────────────────────────
        let status_line = if app.loading {
            Line::from(vec![
                Span::styled("  ~ ", Style::default().fg(Color::Rgb(200, 150, 50))),
                Span::styled("Thinking...", Style::default().fg(Color::Rgb(200, 150, 50))),
            ])
        } else {
            let scroll_indicator = if app.scroll_offset.is_some() {
                " · scrolled (End per tornare in fondo)"
            } else {
                ""
            };
            Line::from(vec![
                Span::styled("  ■ ", Style::default().fg(Color::Rgb(80, 200, 120))),
                Span::styled("Dwarf", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::styled(" · ", Style::default().fg(Color::Rgb(60, 60, 80))),
                Span::styled("15M", Style::default().fg(Color::Rgb(80, 80, 100))),
                Span::styled("  ", Style::default()),
                Span::styled(&app.status, Style::default().fg(Color::Rgb(60, 60, 80))),
                Span::styled(scroll_indicator, Style::default().fg(Color::Rgb(200, 150, 50))),
            ])
        };
        f.render_widget(
            Paragraph::new(status_line).style(Style::default().bg(bg_color)),
            main_layout[2],
        );
    }

    max_scroll
}

fn render_input_bar(f: &mut ratatui::Frame, app: &App, area: ratatui::prelude::Rect, bg_color: Color) {
    let is_editing = app.input_mode == InputMode::Editing;
    let border_color = if is_editing {
        Color::Rgb(100, 149, 237) // Blue when editing
    } else {
        Color::Rgb(40, 40, 60)
    };

    let cursor = if is_editing { "█" } else { "" };
    let input_text = format!("{}{}", app.input, cursor);

    let input_para = Paragraph::new(Line::from(vec![
        Span::styled("  ", Style::default()),
        Span::styled(&input_text, Style::default().fg(Color::White)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .style(Style::default().bg(bg_color)),
    )
    .style(Style::default().bg(bg_color));

    f.render_widget(input_para, area);
}