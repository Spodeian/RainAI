use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Tabs},
    Terminal,
};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, BufReader},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};
use sysinfo::{Pid, Signal, System};

struct App {
    pub active_tab: usize,
    pub logs: Vec<String>,
    pub surface_stats: HashMap<String, usize>,
    pub total_chunks: usize,
    pub loss_history: Vec<u64>,
    pub sys: System,
    pub cpu_usage: f64,
    pub mem_usage: f64,
    
    pub active_pid: Option<u32>,
    pub active_task_name: Option<String>,
    
    log_rx: Receiver<String>,
    pub log_tx: Sender<String>,
    last_tick: Instant,
}

impl App {
    fn new() -> App {
        let (tx, rx) = mpsc::channel();
        let mut sys = System::new_all();
        sys.refresh_all();

        let mut app = App {
            active_tab: 0,
            logs: vec![String::from("RainAI Terminal Studio Initialized.")],
            surface_stats: HashMap::new(),
            total_chunks: 0,
            loss_history: vec![],
            sys,
            cpu_usage: 0.0,
            mem_usage: 0.0,
            active_pid: None,
            active_task_name: None,
            log_rx: rx,
            log_tx: tx,
            last_tick: Instant::now(),
        };
        app.refresh_manifest();
        app
    }

    fn refresh_manifest(&mut self) {
        self.surface_stats.clear();
        self.total_chunks = 0;

        let manifest_paths = [
            "Data/processed/manifest.json",
            "data/processed/manifest.json",
            "../Data/processed/manifest.json",
        ];

        let mut content = String::new();
        for path in manifest_paths {
            if Path::new(path).exists() {
                if let Ok(c) = fs::read_to_string(path) {
                    content = c;
                    break;
                }
            }
        }

        if !content.is_empty() {
            if let Ok(parsed) = serde_json::from_str::<HashMap<String, Value>>(&content) {
                self.total_chunks = parsed.len();
                for (_, meta) in parsed {
                    if let Some(tag) = meta.get("surface_tag").and_then(|t| t.as_str()) {
                        *self.surface_stats.entry(tag.to_string()).or_insert(0) += 1;
                    }
                }
                let _ = self.log_tx.send(format!("[+] Manifest refreshed: {} chunks active.", self.total_chunks));
            }
        } else {
            let _ = self.log_tx.send("[!] Manifest not found. Run Feature Extraction or Heal Corpus (Tab 2).".to_string());
        }
    }

    fn prune_data(&mut self) {
        let tx = self.log_tx.clone();
        thread::spawn(move || {
            let _ = tx.send("[*] Pruning processed datasets and synthetic temp files...".into());
            let _ = fs::remove_dir_all("Data/processed");
            let _ = fs::remove_dir_all("data/processed");
            let _ = fs::remove_dir_all("Data/rain/Synthetic");
            let _ = tx.send("[+] Pruning complete. Corpus ready for fresh ingestion/preparation.".into());
        });
    }

    fn audit_attributions(&mut self) {
        let tx = self.log_tx.clone();
        thread::spawn(move || {
            let attr_path = Path::new("Data/rain/ATTRIBUTIONS.txt");
            if attr_path.exists() {
                if let Ok(content) = fs::read_to_string(attr_path) {
                    let lines: Vec<&str> = content.lines().collect();
                    let _ = tx.send(format!("[+] Attribution Audit: Found {} verified commercial-compliant assets.", lines.len()));
                    for line in lines.iter().take(5) {
                        let _ = tx.send(format!("    -> {}", line));
                    }
                    if lines.len() > 5 {
                        let _ = tx.send(format!("    ... and {} more entries in ATTRIBUTIONS.txt", lines.len() - 5));
                    }
                }
            } else {
                let _ = tx.send("[!] ATTRIBUTIONS.txt not found. Run Multi-Source Ingestion first.".into());
            }
        });
    }

    fn dispatch_process(&mut self, cmd: &str, args: Vec<&'static str>, description: &'static str) {
        if let Some(current_task) = &self.active_task_name {
            let _ = self.log_tx.send(format!("[!] Cannot launch '{}'. Process '{}' is currently active.", description, current_task));
            return;
        }

        let tx = self.log_tx.clone();
        let _ = tx.send("================================================".to_string());
        let _ = tx.send(format!("[*] Launching: {} {:?} ({})", cmd, args, description));

        let cmd_string = cmd.to_string();
        let args_owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let desc_owned = description.to_string();

        match Command::new(&cmd_string)
            .args(&args_owned)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                let pid = child.id();
                self.active_pid = Some(pid);
                self.active_task_name = Some(desc_owned.clone());
                
                let tx_clone = tx.clone();
                let stdout = child.stdout.take().unwrap();
                let stderr = child.stderr.take().unwrap();

                thread::spawn(move || {
                    let tx_out = tx_clone.clone();
                    let _out_thread = thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines().map_while(Result::ok) {
                            let _ = tx_out.send(line);
                        }
                    });

                    let tx_err = tx_clone.clone();
                    let _err_thread = thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines().map_while(Result::ok) {
                            let _ = tx_err.send(format!("[ERR] {}", line));
                        }
                    });

                    match child.wait() {
                        Ok(status) if status.success() => {
                            let _ = tx_clone.send(format!("[+] Task '{}' completed successfully.", desc_owned));
                        }
                        Ok(status) => {
                            let _ = tx_clone.send(format!("[!] Task '{}' exited with status: {}", desc_owned, status));
                        }
                        Err(e) => {
                            let _ = tx_clone.send(format!("[!] Task '{}' failed execution: {}", desc_owned, e));
                        }
                    }
                    
                    let _ = tx_clone.send(format!("TASK_COMPLETE:{}", pid));
                });
            }
            Err(e) => {
                let _ = tx.send(format!("[!] Failed to spawn process: {}", e));
            }
        };
    }

    fn cancel_active_process(&mut self) {
        if let Some(pid) = self.active_pid {
            self.sys.refresh_processes();
            if let Some(process) = self.sys.process(Pid::from_u32(pid)) {
                let _ = self.log_tx.send(format!("[!] Sending Interrupt (SIGINT) to PID {}...", pid));
                let sent = process.kill_with(Signal::Interrupt).unwrap_or(false);
                if !sent {
                    let _ = self.log_tx.send(format!("[!] SIGINT unhandled. Forcibly terminating PID {}...", pid));
                    process.kill(); 
                }
            } else {
                let _ = self.log_tx.send("[!] Active process no longer found in system table.".to_string());
            }
        }
    }

    fn poll_channels(&mut self) {
        if self.last_tick.elapsed() >= Duration::from_secs(1) {
            self.sys.refresh_cpu_usage();
            self.sys.refresh_memory();
            self.cpu_usage = self.sys.global_cpu_info().cpu_usage() as f64;
            
            let total_mem = self.sys.total_memory() as f64;
            let used_mem = self.sys.used_memory() as f64;
            self.mem_usage = if total_mem > 0.0 { (used_mem / total_mem) * 100.0 } else { 0.0 };
            
            self.last_tick = Instant::now();
        }

        while let Ok(msg) = self.log_rx.try_recv() {
            if msg.starts_with("TASK_COMPLETE:") {
                self.active_pid = None;
                self.active_task_name = None;
                continue;
            }

            if msg.contains("Train Loss:") {
                if let Some(loss_str) = msg.split("Train Loss:").nth(1).and_then(|s| s.split('|').next()) {
                    if let Ok(loss_val) = loss_str.trim().parse::<f32>() {
                        let visual_val = (loss_val.clamp(0.0, 1.0) * 100.0) as u64;
                        self.loss_history.push(visual_val);
                        if self.loss_history.len() > 150 {
                            self.loss_history.remove(0);
                        }
                    }
                }
            }

            self.logs.push(msg);
            if self.logs.len() > 350 {
                self.logs.remove(0);
            }
        }
    }
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let app = App::new();
    let res = run_app(&mut terminal, app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err);
    }
    Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
    loop {
        app.poll_channels();
        terminal.draw(|f| ui(f, &app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('x') => app.cancel_active_process(),
                    KeyCode::Char('1') => app.active_tab = 0,
                    KeyCode::Char('2') => app.active_tab = 1,
                    KeyCode::Char('3') => app.active_tab = 2,
                    KeyCode::Char('4') => app.active_tab = 3,
                    KeyCode::Char('5') => app.active_tab = 4,
                    KeyCode::Char('r') => app.refresh_manifest(),
                    
                    // Tab 2: Data Pipeline & Multi-Source Management
                    KeyCode::Char('p') if app.active_tab == 1 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "export-only", "--prepare-data"], "Prepare Full Dataset"),
                    KeyCode::Char('h') if app.active_tab == 1 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "export-only", "--rebuild-data"], "Heal & Rebuild Corpus"),
                    KeyCode::Char('a') if app.active_tab == 1 => app.audit_attributions(),
                    KeyCode::Char('c') if app.active_tab == 1 => app.prune_data(),
                    KeyCode::Char('i') if app.active_tab == 1 => app.dispatch_process("cargo", vec!["run", "--release", "--bin", "rainai_ingest"], "Ingest Multi-Source Audio (Freesound, Mixkit, etc.)"),
                    KeyCode::Char('s') if app.active_tab == 1 => app.dispatch_process("cargo", vec!["run", "--release", "--bin", "rainai_synth"], "Synthesize Synthetic Rain (Gunn-Kinzer)"),
                    KeyCode::Char('u') if app.active_tab == 1 => app.dispatch_process("cargo", vec!["run", "--release", "--bin", "rainai_upmix"], "Spatial Upmixing (Mono -> 48kHz FOA)"),
                    KeyCode::Char('f') if app.active_tab == 1 => app.dispatch_process("cargo", vec!["run", "--release", "--bin", "rainai_features"], "Feature Extraction & Manifest Sync"),
                    
                    // Tab 3: Extended Training Profiles & Granular Stages
                    KeyCode::Char('t') if app.active_tab == 2 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "smoke-test"], "Smoke Test Profile (Fast Check)"),
                    KeyCode::Char('b') if app.active_tab == 2 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "balanced"], "Balanced Training Profile (3 VAE, 2 Mamba)"),
                    KeyCode::Char('p') if app.active_tab == 2 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "production"], "Full Production Profile (10 VAE, 5 Mamba, Disc)"),
                    KeyCode::Char('e') if app.active_tab == 2 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "export-only"], "Export-Only Profile (Skip Training)"),
                    KeyCode::Char('v') if app.active_tab == 2 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "custom", "--phases", "vae"], "Train Spatial VAE Only (Compiled)"),
                    KeyCode::Char('m') if app.active_tab == 2 => app.dispatch_process("python", vec!["scripts/auto_train.py", "--profile", "custom", "--phases", "mamba"], "Train Mamba-2 MoE Only (Compiled)"),
                    
                    // Tab 4: Export & Benchmark
                    KeyCode::Char('e') if app.active_tab == 3 => app.dispatch_process("python", vec!["scripts/export_all.py"], "Full Multi-Backend Artifact Export"),
                    KeyCode::Char('b') if app.active_tab == 3 => app.dispatch_process("cargo", vec!["bench"], "Native Criterion Benchmarks"),

                    _ => {}
                }
            }
        }
    }
}

fn ui(f: &mut ratatui::Frame, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ].as_ref())
        .split(f.size());

    let tab_titles: Vec<Line> = vec!["1: Corpus", "2: Data Pipeline", "3: Training Studio", "4: Export & Bench", "5: Live Logs"]
        .into_iter()
        .map(|t| Line::from(Span::styled(t, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))))
        .collect();

    let tabs = Tabs::new(tab_titles)
        .block(Block::default().borders(Borders::ALL).title(" RainAI Terminal Studio "))
        .select(app.active_tab)
        .style(Style::default().fg(Color::Cyan))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED).fg(Color::White));
    f.render_widget(tabs, root[0]);

    let status_text = match &app.active_task_name {
        Some(task) => Span::styled(format!(" [RUNNING] Active Task: {} (Press 'x' to abort) ", task), Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD)),
        None => Span::styled(" [IDLE] System ready. ", Style::default().fg(Color::Gray)),
    };
    let status_para = Paragraph::new(Line::from(status_text)).block(Block::default().borders(Borders::ALL));
    f.render_widget(status_para, root[1]);

    match app.active_tab {
        0 => render_dashboard(f, app, root[2]),
        1 => render_data_tasks(f, root[2]),
        2 => render_training_studio(f, app, root[2]),
        3 => render_export_bench(f, root[2]),
        4 => render_logs(f, app, root[2]),
        _ => {}
    }

    render_telemetry_footer(f, app, root[3]);
}

fn render_dashboard(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(area);

    let summary_text = vec![
        Line::from(vec![Span::styled("Total FOA Chunks: ", Style::default().fg(Color::Yellow)), Span::raw(app.total_chunks.to_string())]),
        Line::from(vec![Span::styled("Total Duration:   ", Style::default().fg(Color::Yellow)), Span::raw(format!("{:.1} mins", (app.total_chunks as f32 * 5.0) / 60.0))]),
        Line::from(""),
        Line::from(Span::styled("Press 'r' to reload active manifest states.", Style::default().fg(Color::DarkGray))),
    ];

    let summary_para = Paragraph::new(summary_text)
        .block(Block::default().title(" Corpus Summary ").borders(Borders::ALL));
    f.render_widget(summary_para, chunks[0]);

    let mut sorted_stats: Vec<(&String, &usize)> = app.surface_stats.iter().collect();
    sorted_stats.sort_by(|a, b| b.1.cmp(a.1));

    let list_items: Vec<ListItem> = sorted_stats.iter()
        .map(|(k, v)| {
            let pct = if app.total_chunks > 0 { (**v as f32 / app.total_chunks as f32) * 100.0 } else { 0.0 };
            ListItem::new(format!("{:<16} | {:>4} chunks ({:>5.1}%)", k, v, pct))
        })
        .collect();
    
    let surfaces_list = List::new(list_items).block(Block::default().title(" Surface Distribution ").borders(Borders::ALL));
    f.render_widget(surfaces_list, chunks[1]);
}

fn render_data_tasks(f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
    let tasks = vec![
        ListItem::new(Line::from(vec![Span::styled("[p]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" Prepare Dataset (Ingest + Synth)      => auto_train.py --prepare-data")])),
        ListItem::new(Line::from(vec![Span::styled("[h]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" Heal / Rebuild Corpus & Manifest       => auto_train.py --rebuild-data")])),
        ListItem::new(Line::from(vec![Span::styled("[a]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)), Span::raw(" Audit Attributions (ATTRIBUTIONS.txt)  => (Memory Inspection)")])),
        ListItem::new(Line::from(vec![Span::styled("[c]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)), Span::raw(" Clean / Prune Processed Data           => (Native File System)")])),
        ListItem::new(Line::from(" ")),
        ListItem::new(Line::from(vec![Span::styled("[i]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" Ingest Multi-Source Open Audio         => rainai_ingest (Freesound, Mixkit, etc.)")])),
        ListItem::new(Line::from(vec![Span::styled("[s]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" Synthesize Rain (Gunn-Kinzer DSD)      => rainai_synth")])),
        ListItem::new(Line::from(vec![Span::styled("[u]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" Spatial Upmixing (Mono -> 48kHz FOA)   => rainai_upmix")])),
        ListItem::new(Line::from(vec![Span::styled("[f]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" Feature Extraction (RMS, Transients)   => rainai_features")])),
    ];

    let task_list = List::new(tasks).block(Block::default().title(" Multi-Source Data Pipeline & Management ").borders(Borders::ALL));
    f.render_widget(task_list, area);
}

fn render_training_studio(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(0)].as_ref())
        .split(area);

    let tasks = vec![
        ListItem::new(Line::from(vec![Span::styled("[t]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)), Span::raw(" Smoke Test Profile (1 VAE, 1 Mamba, fast check)")])),
        ListItem::new(Line::from(vec![Span::styled("[b]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)), Span::raw(" Balanced Profile (3 VAE, 2 Mamba, AMP) [Default]")])),
        ListItem::new(Line::from(vec![Span::styled("[p]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)), Span::raw(" Production Profile (10 VAE, 5 Mamba, STFT Disc)")])),
        ListItem::new(Line::from(vec![Span::styled("[e]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)), Span::raw(" Export-Only Profile (Skip Training, Build S0-S4)")])),
        ListItem::new(Line::from(" ")),
        ListItem::new(Line::from(vec![Span::styled("[v]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)), Span::raw(" Train Spatial VAE Only (Phase 2 Compiled)")])),
        ListItem::new(Line::from(vec![Span::styled("[m]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)), Span::raw(" Train Mamba-2 MoE Only (Phase 3 Compiled)")])),
    ];
    let task_list = List::new(tasks).block(Block::default().title(" Extended Training Profiles & Stages ").borders(Borders::ALL));
    f.render_widget(task_list, chunks[0]);

    let sparkline = Sparkline::default()
        .block(Block::default().title(" Real-time Loss Trajectory ").borders(Borders::ALL))
        .data(&app.loss_history)
        .style(Style::default().fg(Color::Red));
    f.render_widget(sparkline, chunks[1]);
}

fn render_export_bench(f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
    let tasks = vec![
        ListItem::new(Line::from(vec![Span::styled("[e]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)), Span::raw(" Export Artifacts (SafeTensors, ONNX, S0-S4)")])),
        ListItem::new(Line::from(vec![Span::styled("[b]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)), Span::raw(" Run Native Criterion Benchmarks")])),
    ];
    let task_list = List::new(tasks).block(Block::default().title(" Production Verification Phase ").borders(Borders::ALL));
    f.render_widget(task_list, area);
}

fn render_logs(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let log_text: Vec<Line> = app.logs.iter().map(|l| Line::from(l.as_str())).collect();
    let logs_para = Paragraph::new(log_text)
        .block(Block::default().title(" Standard Output / Error Stream ").borders(Borders::ALL))
        .scroll(((app.logs.len().saturating_sub(area.height as usize - 2)) as u16, 0));
    f.render_widget(logs_para, area);
}

fn render_telemetry_footer(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(area);

    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" System CPU "))
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
        .ratio((app.cpu_usage / 100.0).clamp(0.0, 1.0));
    f.render_widget(cpu_gauge, chunks[0]);

    let mem_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" System RAM "))
        .gauge_style(Style::default().fg(Color::Magenta).bg(Color::Black))
        .ratio((app.mem_usage / 100.0).clamp(0.0, 1.0));
    f.render_widget(mem_gauge, chunks[1]);
}
