#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]
//! RainAI Graphical User Interface & Studio View Controllers.
//!
//! Provides the cross-platform immediate-mode UI powered by `egui` and `eframe`.
//! Features real-time 3D spatial radar visualizations, 16-band interactive FFT spectrograms,
//! acoustic surface material matrix sliders, hardware stress indicators, and preset managers.
//!
//! # Architecture & Modules
//!
//! - [`components`]: Modular UI panels including rain controls, 3D Ambisonic radar, real-time
//!   spectrogram, governor telemetry monitors, and audio export dialogues.
//! - [`storage_manager`]: Synchronous/asynchronous persistence bridge saving and loading
//!   user configurations from browser `localStorage` or native filesystem directories.

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
pub use spodeian_ui::ScreenConstraints;

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
    pub last_wake_lock_state: bool,
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
            last_wake_lock_state: false,
        }
    }
}

impl TemplateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing RainAI Studio...");

        let mut loaded_from_storage = true;
        #[allow(unused_mut)]
        let mut state = load_state_multi_tier(cc.storage).unwrap_or_else(|| {
            warn!("No saved state found in storage, initializing fresh defaults.");
            loaded_from_storage = false;
            AppState::default()
        });

        #[cfg(target_arch = "wasm32")]
        {
            if !loaded_from_storage {
                if let Some(win) = web_sys::window() {
                    let inner_w = win.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(1024.0);
                    if inner_w < 650.0 {
                        info!("Detected mobile viewport ({:.0}px), defaulting fresh session to EcoBatterySaver profile", inner_w);
                        state.rain.optimization_profile = shared::GovernorOptimizationProfile::EcoBatterySaver;
                        state.rain.thinking_steps = 1;
                        state.rain.use_consistency_jump = true;
                    }
                }
            }

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
        spodeian_ui::apply_theme(ctx, self.state.config.theme);
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

    pub fn sync_audio_engine(&mut self) {
        self.ensure_audio_engine();

        if let Some(ref audio_state) = self.audio_state {
            audio_state.update_rain(&self.state.rain);
            audio_state.set_decode_mode(self.rain_view.decode_mode);
            audio_state.set_orientation(self.rain_view.listener_yaw, 0.0, 0.0);
            self.state.rain.telemetry = audio_state.get_telemetry();

            let mut current_bins = [0.0f32; 32];
            for (i, bin) in current_bins.iter_mut().enumerate() {
                let intensity = (self.state.rain.weather.intensity * 0.7 
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

            if self.state.rain.is_playing != self.last_wake_lock_state {
                self.last_wake_lock_state = self.state.rain.is_playing;
                crate::storage_manager::set_screen_wake_lock(self.last_wake_lock_state);
            }
        }
    }
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

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.apply_theme(ctx);
        
        let egui_theme = match self.state.config.theme {
            ThemeMode::Light | ThemeMode::HighContrastLight => egui::Theme::Light,
            ThemeMode::Dark | ThemeMode::HighContrastDark => egui::Theme::Dark,
        };

        if self.state.config.theme.is_high_contrast() {
            ctx.style_mut_of(egui_theme, |style| {
                style.spacing.interact_size = egui::vec2(44.0, 44.0);
                style.spacing.button_padding = egui::vec2(14.0, 10.0);
            });
        } else {
            ctx.style_mut_of(egui_theme, |style| {
                style.spacing.interact_size.y = style.spacing.interact_size.y.max(32.0);
                style.spacing.button_padding = egui::vec2(12.0, 8.0);
            });
        }

        self.handle_keyboard_shortcuts(ctx);
        self.sync_audio_engine();

        let cur_time = ctx.input(|i| i.time);
        if cur_time - self.last_diag_poll_time > 2.0 {
            self.last_diag_poll_time = cur_time;
            let queried = query_storage_diagnostics();
            self.storage_diag.is_persisted = queried.is_persisted;
            self.storage_diag.pwa_install_available = queried.pwa_install_available;
            self.storage_diag.is_pwa_installed = queried.is_pwa_installed;
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let constraints = ScreenConstraints::compute(ui);

        // 1. Render navbar (Top Panel)
        components::navbar::render_navbar(self, ui, &constraints);

        // 2. Render bottom spectrogram waterfall panel
        render_spectrogram_panel(ui, &self.spectrogram_history);

        // 3. Central content area
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.group(|ui| {
                    ui.heading("🌧 RainAI Neural Spatial Soundscape Studio");
                    ui.label("Continuous, non-repetitive procedural rain synthesis conditioned on 554 physical parameters.");
                    ui.add_space(8.0);
                    self.rain_view.render(ui, &mut self.state.rain, self.audio_state.as_ref());
                });
            });
        });

        // 4. Modals and warnings
        components::modals::render_dialogs(self, ui);
        components::modals::render_warning_banners(self, ui.ctx());
    }
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use eframe::NativeOptions;
    let mut options = NativeOptions::default();
    options.android_app = Some(app);
    eframe::run_native(
        "RainAI",
        options,
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
    ).unwrap();
}

