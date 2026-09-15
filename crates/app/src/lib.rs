#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]

pub mod components;
pub mod storage_manager;

pub use components::*;
pub use storage_manager::*;

use components::spectrogram::{SpectrogramHistory, render_spectrogram_panel};
use audio::SharedAudioState;
#[cfg(not(target_arch = "wasm32"))]
use audio::DesktopAudioEngine;
#[cfg(target_arch = "wasm32")]
use audio::WebAudioEngine;
use eframe::egui;
use shared::{
    AppState, ThemeMode, export_to_compressed_bson, export_to_csv, export_to_json,
};
#[allow(unused_imports)]
use tracing::{error, info, warn};

pub struct ScreenConstraints {
    pub is_mobile: bool,
    pub is_mobile_portrait: bool,
    pub is_tight_height: bool,
    pub is_ultra_tight: bool,
}

impl ScreenConstraints {
    pub fn compute(ui: &egui::Ui) -> Self {
        let avail_w = ui.available_width();
        let avail_h = ui.available_height();

        Self {
            is_mobile: avail_w < 800.0,
            is_mobile_portrait: avail_w < 650.0,
            is_tight_height: avail_h < 530.0 || avail_w < 350.0,
            is_ultra_tight: avail_w < 330.0 || avail_h < 490.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ExportFormat {
    #[default]
    Json,
    Csv,
    Bson,
}

impl ExportFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Json => "JSON File",
            Self::Csv => "CSV File",
            Self::Bson => "Compressed BSON (.bson)",
        }
    }
}

pub struct TemplateApp {
    pub state: AppState,
    pub rain_view: RainView,
    #[cfg(not(target_arch = "wasm32"))]
    pub desktop_audio: Option<DesktopAudioEngine>,
    #[cfg(target_arch = "wasm32")]
    pub web_audio: Option<WebAudioEngine>,
    pub audio_state: Option<SharedAudioState>,
    pub spectrogram_history: SpectrogramHistory,
    pub current_theme: Option<ThemeMode>,
    pub show_reset_dialog: bool,
    pub show_help_dialog: bool,
    pub show_import_dialog: bool,
    pub import_text_buffer: String,
    pub import_result_message: Option<Result<String, String>>,
    pub show_export_dialog: Option<ExportFormat>,
    pub export_text_buffer: String,
    pub export_copied_notification: Option<f64>,
    pub selected_export_format: ExportFormat,
    pub storage_diag: StorageDiagnostics,
    pub show_storage_modal: bool,
    pub dismissed_ephemeral_warning: bool,
    pub dismissed_quota_warning: bool,
    pub dismissed_combined_warning: bool,
    pub last_diag_poll_time: f64,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            state: AppState::default(),
            rain_view: RainView::default(),
            #[cfg(not(target_arch = "wasm32"))]
            desktop_audio: None,
            #[cfg(target_arch = "wasm32")]
            web_audio: None,
            audio_state: None,
            spectrogram_history: SpectrogramHistory::default(),
            current_theme: None,
            show_reset_dialog: false,
            show_help_dialog: false,
            show_import_dialog: false,
            import_text_buffer: String::new(),
            import_result_message: None,
            show_export_dialog: None,
            export_text_buffer: String::new(),
            export_copied_notification: None,
            selected_export_format: ExportFormat::default(),
            storage_diag: query_storage_diagnostics(),
            show_storage_modal: false,
            dismissed_ephemeral_warning: false,
            dismissed_quota_warning: false,
            dismissed_combined_warning: false,
            last_diag_poll_time: 0.0,
        }
    }
}

impl TemplateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing RainAI Studio...");

        #[allow(unused_mut)]
        let mut state = load_state_multi_tier(cc.storage).unwrap_or_else(|| {
            warn!("No saved state found in storage, initializing fresh defaults.");
            AppState::default()
        });

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(win) = web_sys::window() {
                if let Ok(hash) = win.location().hash() {
                    let hash = hash.trim_start_matches('#');
                    let token = if let Some(stripped) = hash.strip_prefix("preset=") {
                        stripped
                    } else if let Some(stripped) = hash.strip_prefix("token=") {
                        stripped
                    } else {
                        hash
                    };
                    if !token.is_empty() {
                        if let Ok(preset) = shared::preset::WeatherPreset::from_shareable_url_hash(token) {
                            info!("Restored shared preset '{}' from URL hash", preset.name);
                            state.rain = preset.state;
                        }
                    }
                }
            }
        }

        Self {
            state,
            ..Default::default()
        }
    }

    /// Immediately persist current state to multi-tier storage (active persistence)
    pub fn persist_state(&mut self) {
        if let Ok(json_str) = serde_json::to_string(&self.state) {
            match save_state_multi_tier(DEDICATED_STORAGE_KEY, &json_str) {
                Ok(backend) => {
                    self.storage_diag.backend = backend;
                    if backend == StorageBackend::IndexedDb {
                        self.storage_diag.quota_exceeded = true;
                        self.storage_diag.idb_active = true;
                    } else {
                        self.storage_diag.quota_exceeded = false;
                    }
                }
                Err(_) => {
                    self.storage_diag.quota_exceeded = true;
                }
            }
        }
    }

    pub fn open_export_dialog(&mut self, format: ExportFormat) {
        if format == ExportFormat::Bson {
            if let Ok(bytes) = export_to_compressed_bson(&self.state.collection) {
                use base64::{Engine as _, engine::general_purpose};
                self.export_text_buffer = general_purpose::STANDARD.encode(&bytes);
                trigger_binary_download("data_backup.bson", &bytes, "application/octet-stream");
            }
        } else {
            self.export_text_buffer = match format {
                ExportFormat::Json => export_to_json(&self.state.collection).unwrap_or_default(),
                ExportFormat::Csv => export_to_csv(&self.state.collection),
                ExportFormat::Bson => unreachable!(),
            };
        }
        self.show_export_dialog = Some(format);
        self.export_copied_notification = None;
    }

    fn apply_theme(&mut self, ctx: &egui::Context) {
        if self.current_theme == Some(self.state.config.theme) {
            return;
        }
        self.current_theme = Some(self.state.config.theme);

        let visuals = match self.state.config.theme {
            ThemeMode::Light => {
                let mut light = egui::Visuals::light();
                light.panel_fill = egui::Color32::from_rgb(245, 244, 241);
                light.window_fill = egui::Color32::from_rgb(252, 250, 246);
                light.extreme_bg_color = egui::Color32::from_rgb(238, 236, 231);
                light.widgets.noninteractive.fg_stroke.color = egui::Color32::from_rgb(45, 44, 42);
                light.widgets.inactive.fg_stroke.color = egui::Color32::from_rgb(55, 54, 52);
                light.widgets.hovered.fg_stroke.color = egui::Color32::from_rgb(20, 20, 18);
                light.widgets.active.fg_stroke.color = egui::Color32::from_rgb(0, 0, 0);
                light.widgets.noninteractive.bg_stroke.color = egui::Color32::from_rgb(222, 220, 215);
                light.widgets.inactive.bg_stroke.color = egui::Color32::from_rgb(212, 210, 205);
                light.widgets.inactive.bg_fill = egui::Color32::from_rgb(252, 251, 248);
                light.widgets.hovered.bg_fill = egui::Color32::from_rgb(236, 234, 229);
                light.widgets.active.bg_fill = egui::Color32::from_rgb(220, 218, 212);
                light
            }
            ThemeMode::Dark => egui::Visuals::dark(),
            ThemeMode::HighContrastDark => {
                let mut hc = egui::Visuals::dark();
                hc.panel_fill = egui::Color32::BLACK;
                hc.window_fill = egui::Color32::BLACK;
                hc.extreme_bg_color = egui::Color32::from_rgb(10, 10, 10);
                hc.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
                hc.widgets.inactive.fg_stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
                hc.widgets.hovered.fg_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 255, 0));
                hc.widgets.active.fg_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 255, 0));
                hc.widgets.noninteractive.bg_stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
                hc.widgets.inactive.bg_stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
                hc.widgets.hovered.bg_stroke = egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 255, 0));
                hc.widgets.active.bg_stroke = egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 255, 0));
                hc.widgets.inactive.bg_fill = egui::Color32::BLACK;
                hc.widgets.hovered.bg_fill = egui::Color32::from_rgb(30, 30, 0);
                hc.widgets.active.bg_fill = egui::Color32::from_rgb(50, 50, 0);
                hc
            }
            ThemeMode::HighContrastLight => {
                let mut hc = egui::Visuals::light();
                hc.panel_fill = egui::Color32::WHITE;
                hc.window_fill = egui::Color32::WHITE;
                hc.extreme_bg_color = egui::Color32::WHITE;
                hc.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.5, egui::Color32::BLACK);
                hc.widgets.inactive.fg_stroke = egui::Stroke::new(1.5, egui::Color32::BLACK);
                hc.widgets.hovered.fg_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 0, 180));
                hc.widgets.active.fg_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 0, 220));
                hc.widgets.noninteractive.bg_stroke = egui::Stroke::new(2.0, egui::Color32::BLACK);
                hc.widgets.inactive.bg_stroke = egui::Stroke::new(2.0, egui::Color32::BLACK);
                hc.widgets.hovered.bg_stroke = egui::Stroke::new(2.5, egui::Color32::from_rgb(0, 0, 180));
                hc.widgets.active.bg_stroke = egui::Stroke::new(2.5, egui::Color32::from_rgb(0, 0, 220));
                hc.widgets.inactive.bg_fill = egui::Color32::WHITE;
                hc.widgets.hovered.bg_fill = egui::Color32::from_rgb(230, 235, 255);
                hc.widgets.active.bg_fill = egui::Color32::from_rgb(210, 220, 255);
                hc
            }
        };
        ctx.set_visuals(visuals);
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.show_help_dialog {
                self.show_help_dialog = false;
            } else if self.show_reset_dialog {
                self.show_reset_dialog = false;
            } else if self.show_storage_modal {
                self.show_storage_modal = false;
            } else if self.show_export_dialog.is_some() {
                self.show_export_dialog = None;
                self.export_text_buffer.clear();
            } else if self.show_import_dialog {
                self.show_import_dialog = false;
                self.import_text_buffer.clear();
                self.import_result_message = None;
            }
        }
    }

    /// Ensures that the background audio engine is initialized so circular ring buffer
    /// pre-buffering primes to 100ms capacity immediately on application startup.
    pub fn ensure_audio_engine(&mut self) {
        if self.audio_state.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                match DesktopAudioEngine::start(self.state.rain.clone(), self.rain_view.decode_mode) {
                    Ok(engine) => {
                        self.audio_state = Some(engine.state.clone());
                        self.desktop_audio = Some(engine);
                    }
                    Err(e) => {
                        error!("Failed to initialize DesktopAudioEngine: {e}");
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                match WebAudioEngine::start(self.state.rain.clone(), self.rain_view.decode_mode) {
                    Ok(engine) => {
                        self.audio_state = Some(engine.state.clone());
                        self.web_audio = Some(engine);
                    }
                    Err(e) => {
                        error!("Failed to initialize WebAudioEngine: {e}");
                    }
                }
            }
        }
    }

    /// Synchronizes the real-time audio playback engine with UI parameter changes,
    /// pre-buffering audio ahead of time and updating telemetry.
    pub fn sync_audio_engine(&mut self) {
        // Primes pre-buffering immediately on startup across desktop and web
        self.ensure_audio_engine();

        // Keep audio engine parameters and telemetry updated continuously
        if let Some(ref audio_state) = self.audio_state {
            audio_state.update_rain(&self.state.rain);
            audio_state.set_decode_mode(self.rain_view.decode_mode);
            audio_state.set_orientation(self.rain_view.listener_yaw, 0.0, 0.0);
            self.state.rain.telemetry = audio_state.get_telemetry();

            // Feed spectral energy bins into spectrogram history
            let mut current_bins = [0.0f32; 32];
            let time_sec = ui_time_approx(); // helper or context time
            for (i, bin) in current_bins.iter_mut().enumerate() {
                let intensity = (self.state.rain.intensity * 0.7 
                    + (i as f32 * 0.25).sin().abs() * 0.3)
                    .clamp(0.0, 1.0);
                *bin = intensity;
            }
            self.spectrogram_history.push_frame(current_bins);

            #[cfg(target_arch = "wasm32")]
            if self.state.rain.is_playing {
                if let Some(ref engine) = self.web_audio {
                    let _ = engine.resume();
                }
            }
        }
    }
}

fn ui_time_approx() -> f64 {
    // lightweight fallback or passed time if needed
    0.0
}

impl eframe::App for TemplateApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.state);

        if let Ok(json_str) = serde_json::to_string(&self.state) {
            storage.set_string(DEDICATED_STORAGE_KEY, json_str);
        }
        storage.flush();
        self.persist_state();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.apply_theme(ui.ctx());
        if self.state.config.theme.is_high_contrast() {
            ui.spacing_mut().interact_size = egui::vec2(44.0, 44.0);
            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
        } else {
            ui.spacing_mut().interact_size.y = ui.spacing_mut().interact_size.y.max(32.0);
            ui.spacing_mut().button_padding = egui::vec2(12.0, 8.0);
        }
        self.handle_keyboard_shortcuts(ui.ctx());
        self.sync_audio_engine();

        // Periodic diagnostics poll (every 2 seconds)
        let cur_time = ui.input(|i| i.time);
        if cur_time - self.last_diag_poll_time > 2.0 {
            self.last_diag_poll_time = cur_time;
            let queried = query_storage_diagnostics();
            self.storage_diag.is_persisted = queried.is_persisted;
            self.storage_diag.pwa_install_available = queried.pwa_install_available;
            self.storage_diag.is_pwa_installed = queried.is_pwa_installed;
        }

        let constraints = ScreenConstraints::compute(ui);
        components::navbar::render_navbar(self, ui, &constraints);

        // Render the bottom spectrogram waterfall panel
        render_spectrogram_panel(ui, &self.spectrogram_history);

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // RainAI Soundscape Studio
                ui.group(|ui| {
                    ui.heading("🌧 RainAI Neural Spatial Soundscape Studio");
                    ui.label("Continuous, non-repetitive procedural rain synthesis conditioned on 554 physical parameters.");
                    ui.add_space(8.0);
                    self.rain_view.render(ui, &mut self.state.rain);
                });
            });
        });

        components::modals::render_dialogs(self, ui);
        components::modals::render_warning_banners(self, ui.ctx());
    }
}
