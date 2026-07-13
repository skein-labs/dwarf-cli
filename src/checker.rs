use anyhow::Result;
use std::path::Path;
use std::process::Command;

pub struct CheckResult {
    pub tool: String,
    pub passed: bool,
    pub messages: Vec<CheckMessage>,
}

pub struct CheckMessage {
    pub level: Level,
    pub line: Option<usize>,
    pub text: String,
}

pub enum Level {
    Error,
    Warning,
    Info,
}

impl Level {
    pub fn symbol(&self) -> &str {
        match self {
            Level::Error => "✗",
            Level::Warning => "!",
            Level::Info => "·",
        }
    }
    pub fn color_hint(&self) -> &str {
        match self {
            Level::Error => "red",
            Level::Warning => "yellow",
            Level::Info => "dim",
        }
    }
}

fn check_bash_syntax(path: &Path) -> Result<CheckResult> {
    let output = Command::new("bash").arg("-n").arg(path).output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut messages = Vec::new();

    if output.status.success() {
        messages.push(CheckMessage { level: Level::Info, line: None, text: "Syntax OK".into() });
    } else {
        for line in stderr.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            let (line_num, text) = if let Some(rest) = line.split(": line ").nth(1) {
                let parts: Vec<&str> = rest.splitn(2, ": ").collect();
                (parts[0].parse::<usize>().ok(), parts.get(1).unwrap_or(&rest).to_string())
            } else {
                (None, line.to_string())
            };
            messages.push(CheckMessage { level: Level::Error, line: line_num, text });
        }
    }
    Ok(CheckResult { tool: "bash -n".into(), passed: output.status.success(), messages })
}

fn check_shellcheck(path: &Path) -> Result<Option<CheckResult>> {
    if Command::new("which").arg("shellcheck").output().map(|o| !o.status.success()).unwrap_or(true) {
        return Ok(None);
    }
    let output = Command::new("shellcheck").arg("-f").arg("gcc").arg("-S").arg("warning").arg(path).output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut messages = Vec::new();

    if output.status.success() && stdout.trim().is_empty() {
        messages.push(CheckMessage { level: Level::Info, line: None, text: "No issues found".into() });
    } else {
        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            let parts: Vec<&str> = line.splitn(5, ':').collect();
            let (line_num, level, text) = if parts.len() >= 5 {
                let num = parts[1].trim().parse::<usize>().ok();
                let rest = parts[4].trim();
                let lvl = if rest.starts_with("error") { Level::Error }
                          else if rest.starts_with("warning") { Level::Warning }
                          else { Level::Info };
                let msg = rest.trim_start_matches("error: ").trim_start_matches("warning: ").trim_start_matches("note: ").to_string();
                (num, lvl, msg)
            } else {
                (None, Level::Warning, line.to_string())
            };
            messages.push(CheckMessage { level, line: line_num, text });
        }
    }
    Ok(Some(CheckResult { tool: "shellcheck".into(), passed: output.status.success(), messages }))
}

pub fn check_script_string(script: &str) -> Result<Vec<CheckResult>> {
    let tmp = std::env::temp_dir().join("dwarf_check.sh");
    std::fs::write(&tmp, script)?;
    let results = check_file(&tmp)?;
    let _ = std::fs::remove_file(&tmp);
    Ok(results)
}

pub fn check_file(path: &Path) -> Result<Vec<CheckResult>> {
    let mut results = Vec::new();
    results.push(check_bash_syntax(path)?);
    if let Some(sc) = check_shellcheck(path)? { results.push(sc); }
    Ok(results)
}
