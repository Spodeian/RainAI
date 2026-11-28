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
pub mod settings;
pub mod storage_manager;
pub mod task_queue;

pub use components::*;
pub use settings::*;
pub use storage_manager::*;
pub use task_queue::*;

#[cfg(not(target_arch = "wasm32"))]
use audio::DesktopAudioEngine;
use audio::SharedAudioState;
#[cfg(target_arch = "wasm32")]
use audio::WebAudioEngine;
use components::spectrogram::{SpectrogramHistory, render_spectrogram_panel};
use eframe::egui;
use shared::{AppState, ThemeMode, export_to_compressed_bson, export_to_csv};
pub use spodeian_ui::ScreenConstraints;
#[allow(unused_imports)]
use tracing::{error, info, warn};

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

#[derive(Clone, Debug)]
pub struct ContributeModalState {
    pub submission_mode: usize, // 0: Local File, 1: Remote URL
    pub file_or_url: String,
    pub authors: Vec<String>,
    pub attribution_file_path: String,
    pub selected_license: String,
    pub custom_license_text: String,
    pub custom_license_file_path: String,
    pub tags_input: String,
    pub description_input: String,
    pub description_file_path: String,
    pub confirmed_rights_warranty: bool,
    pub auto_proceed_after_processing: bool,
    pub show_license_details: bool,
    pub acoustic_feedback: Option<String>,
    pub status_message: Option<Result<String, String>>,
}

impl Default for ContributeModalState {
    fn default() -> Self {
        Self {
            submission_mode: 0,
            file_or_url: String::new(),
            authors: vec![String::new()],
            attribution_file_path: String::new(),
            selected_license: "RainAI-FC-Proprietary-License".to_string(),
            custom_license_text: String::new(),
            custom_license_file_path: String::new(),
            tags_input: "tin_roof, rain_texture".to_string(),
            description_input: String::new(),
            description_file_path: String::new(),
            confirmed_rights_warranty: false,
            auto_proceed_after_processing: true,
            show_license_details: false,
            acoustic_feedback: None,
            status_message: None,
        }
    }
}

pub struct TemplateApp {
    pub state: AppState,
    pub rain_view: RainView,
    pub settings: SettingsState,
    pub task_queue: TaskQueue,
    #[cfg(not(target_arch = "wasm32"))]
    pub desktop_audio: Option<DesktopAudioEngine>,
    #[cfg(target_arch = "wasm32")]
    pub web_audio: Option<WebAudioEngine>,
    pub audio_state: Option<SharedAudioState>,
    pub spectrogram_history: SpectrogramHistory,
    pub current_theme: Option<ThemeMode>,
    pub show_reset_dialog: bool,
    pub show_help_dialog: bool,
    pub show_privacy_dialog: bool,
    pub show_settings_dialog: bool,
    pub show_provenance_dialog: bool,
    pub show_presets_dialog: bool,
    pub show_audio_export_dialog: bool,
    pub show_task_queue_tray: bool,
    pub show_contribute_dialog: bool,
    pub contribute_state: ContributeModalState,
    pub preset_state: PresetModalState,
    pub audio_export_state: AudioExportModalState,
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
            settings: SettingsState::default(),
            task_queue: TaskQueue::default(),
            #[cfg(not(target_arch = "wasm32"))]
            desktop_audio: None,
            #[cfg(target_arch = "wasm32")]
            web_audio: None,
            audio_state: None,
            spectrogram_history: SpectrogramHistory::default(),
            current_theme: None,
            show_reset_dialog: false,
            show_help_dialog: false,
            show_privacy_dialog: false,
            show_settings_dialog: false,
            show_provenance_dialog: false,
            show_presets_dialog: false,
            show_audio_export_dialog: false,
            show_task_queue_tray: false,
            show_contribute_dialog: false,
            contribute_state: ContributeModalState::default(),
            preset_state: PresetModalState::default(),
            audio_export_state: AudioExportModalState::default(),
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

        let first_launch = is_first_launch(cc.storage);
        #[allow(unused_variables, unused_mut)]
        let (mut state, loaded_from_storage) = if first_launch {
            info!(
                "First launch detected! Auto-starting Gentle Summer Rain default soundscape at 60% volume."
            );
            let mut rain_state = shared::preset::WeatherPreset::gentle_summer_rain().state;
            rain_state.is_playing = true;
            rain_state.master_volume = 0.60;
            let s = shared::AppState {
                rain: rain_state,
                ..Default::default()
            };
            (s, false)
        } else if let Some(saved) = load_state_multi_tier(cc.storage) {
            (saved, true)
        } else {
            warn!("No saved state found in storage, initializing fresh defaults.");
            (shared::AppState::default(), false)
        };

        let settings = load_settings_state(cc.storage);
        let derived = settings.resolve_parameters();

        let mut rain_view = RainView {
            flow_solver: derived.flow_solver,
            decode_mode: settings.listening_setup.to_decode_mode(),
            webgpu_fp16: true,
            show_advanced_inspector: settings.show_advanced_settings,
            show_spectrogram: settings.show_spectrogram,
            noise_masking_enabled: settings.noise_masking_enabled,
            hrtf_profile: settings.hrtf_profile.clone(),
            ..Default::default()
        };
        state.rain.sound_layers = settings.layers;
        state.rain.evolve_enabled = settings.evolve;
        state.rain.optimization_profile = settings.profile;
        state.rain.quality_tier = derived.waveshaper_tier;

        if first_launch {
            rain_view.toast_notification = Some((
                "Welcome to RainAI! Auto-playing Gentle Summer Rain soundscape.".to_string(),
                25.0,
            ));
        }

        #[allow(unused_variables)]
        if let Some(ir_hash) = &settings.custom_ir_hash {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let cache_dir = std::path::Path::new("data/cache/ir");
                if let Ok(cas) = spodeian_cache::ContentAddressedStorage::new(cache_dir) {
                    if let Ok(bytes) = cas.get(ir_hash) {
                        let _ = rain_view.load_custom_ir_bytes(
                            "cached_ir.wav",
                            &bytes,
                            Some(cache_dir),
                        );
                    }
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            if !loaded_from_storage && !first_launch {
                if let Some(win) = web_sys::window() {
                    let inner_w = win
                        .inner_width()
                        .ok()
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1024.0);
                    if inner_w < 650.0 {
                        info!(
                            "Detected mobile viewport ({:.0}px), defaulting fresh session to EcoBatterySaver profile",
                            inner_w
                        );
                        state.rain.optimization_profile =
                            shared::GovernorOptimizationProfile::EcoBatterySaver;
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
                        if let Ok(preset) =
                            shared::preset::WeatherPreset::from_shareable_url_hash(token)
                        {
                            info!("Restored shared preset '{}' from URL hash", preset.name);
                            state.rain = preset.state;
                        }
                    }
                }
            }
        }

        #[allow(unused_mut)]
        let mut app = Self {
            state,
            rain_view,
            settings,
            ..Default::default()
        };

        #[cfg(not(target_arch = "wasm32"))]
        if first_launch || app.state.rain.is_playing {
            app.ensure_audio_engine();
        }

        // In the browser the IndexedDB copy follows a moment later (see `apply_storage_events`).
        open_web_store(&cc.egui_ctx);
        info!("RainAI Studio ready.");
        app
    }

    /// Applies results from the browser store: the state loaded at startup, and the
    /// outcome of each finished save. Does nothing on native.
    fn apply_storage_events(&mut self) {
        match take_loaded_web_state() {
            Some(Some(mut loaded)) => {
                // As in `new`: the separately saved settings win, and playback stays as the
                // user has it now (browsers need a gesture to start audio anyway).
                loaded.rain.sound_layers = self.settings.layers;
                loaded.rain.evolve_enabled = self.settings.evolve;
                loaded.rain.optimization_profile = self.settings.profile;
                loaded.rain.quality_tier = self.state.rain.quality_tier;
                loaded.rain.is_playing = self.state.rain.is_playing;
                self.state = loaded;
                self.storage_diag.backend = StorageBackend::IndexedDb;
                self.storage_diag.idb_active = true;
            }
            // Nothing in IndexedDB yet: move whatever loaded at startup into it.
            Some(None) => self.persist_state(),
            None => {}
        }
        match take_web_save_outcome() {
            Some(Ok(())) => {
                self.storage_diag.backend = StorageBackend::IndexedDb;
                self.storage_diag.idb_active = true;
                self.storage_diag.quota_exceeded = false;
            }
            Some(Err(message)) => {
                tracing::error!("{message}");
                self.storage_diag.quota_exceeded = message.contains("QuotaExceeded");
            }
            None => {}
        }
    }

    pub fn persist_state(&mut self) {
        self.settings.master_volume = self.state.rain.master_volume;
        self.settings.layers = self.state.rain.sound_layers;
        self.settings.evolve = self.state.rain.evolve_enabled;
        self.settings.profile = self.state.rain.optimization_profile;
        self.settings.show_spectrogram = self.rain_view.show_spectrogram;
        self.settings.show_advanced_settings = self.rain_view.show_advanced_inspector;
        save_settings_state(None, &self.settings);

        let session = PersistentSessionState {
            flow_solver: self.rain_view.flow_solver,
            active_preset_name: self.settings.active_preset_name.clone(),
            master_volume: self.state.rain.master_volume,
            decode_mode: self.rain_view.decode_mode,
            noise_masking_enabled: self.rain_view.noise_masking_enabled,
            noise_masking_threshold_db: -40.0,
            hrtf_profile: self.rain_view.hrtf_profile.clone(),
            webgpu_fp16_enabled: self.rain_view.webgpu_fp16,
            show_advanced_inspector: self.rain_view.show_advanced_inspector,
            show_spectrogram: self.rain_view.show_spectrogram,
            custom_ir_hash: self
                .rain_view
                .custom_ir_meta
                .as_ref()
                .map(|m| m.sha256_hash.clone()),
        };
        save_session_state(None, &session);

        mark_first_launch_done(None);

        if let Ok(json_str) = serde_json::to_string(&self.state) {
            // Browser: queued for IndexedDB; the result arrives in `apply_storage_events`.
            if queue_web_save(&json_str) {
                return;
            }
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
            if let Ok(bytes) = export_to_compressed_bson(&self.state.rain) {
                use base64::{Engine as _, engine::general_purpose};
                self.export_text_buffer = general_purpose::STANDARD.encode(&bytes);
                trigger_binary_download("rain_preset.bson", &bytes, "application/octet-stream");
            }
        } else {
            self.export_text_buffer = match format {
                ExportFormat::Json => {
                    serde_json::to_string_pretty(&self.state.rain).unwrap_or_default()
                }
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
            } else if self.show_contribute_dialog {
                self.show_contribute_dialog = false;
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
            self.rain_view.audio_status_label = "Initializing Audio Engine...".to_string();
            #[cfg(not(target_arch = "wasm32"))]
            {
                match DesktopAudioEngine::start(self.state.rain.clone(), self.rain_view.decode_mode)
                {
                    Ok(engine) => {
                        self.audio_state = Some(engine.state.clone());
                        self.desktop_audio = Some(engine);
                        self.rain_view.audio_status_label =
                            "Ready (48kHz Desktop Audio)".to_string();
                    }
                    Err(e) => {
                        error!("Failed to initialize DesktopAudioEngine: {e}");
                        self.rain_view.audio_status_label = "Audio Fallback Active".to_string();
                        self.rain_view.toast_notification = Some((
                            format!(
                                "Audio engine init error: {e}. Running in visual/procedural fallback mode."
                            ),
                            15.0,
                        ));
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                match WebAudioEngine::start(self.state.rain.clone(), self.rain_view.decode_mode) {
                    Ok(engine) => {
                        self.audio_state = Some(engine.state.clone());
                        self.web_audio = Some(engine);
                        self.rain_view.audio_status_label =
                            "Ready (48kHz WebAudio Spatial)".to_string();
                    }
                    Err(e) => {
                        error!("Failed to initialize WebAudioEngine: {e}");
                        self.rain_view.audio_status_label = "WebAudio Fallback Active".to_string();
                        self.rain_view.toast_notification = Some((
                            format!("WebAudio init error: {e}. Running in visual fallback mode."),
                            15.0,
                        ));
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
        self.settings.master_volume = self.state.rain.master_volume;
        self.settings.layers = self.state.rain.sound_layers;
        self.settings.evolve = self.state.rain.evolve_enabled;
        self.settings.profile = self.state.rain.optimization_profile;
        self.settings.show_spectrogram = self.rain_view.show_spectrogram;
        self.settings.show_advanced_settings = self.rain_view.show_advanced_inspector;
        save_settings_state(Some(storage), &self.settings);

        let session = PersistentSessionState {
            flow_solver: self.rain_view.flow_solver,
            active_preset_name: self.settings.active_preset_name.clone(),
            master_volume: self.state.rain.master_volume,
            decode_mode: self.rain_view.decode_mode,
            noise_masking_enabled: self.rain_view.noise_masking_enabled,
            noise_masking_threshold_db: -40.0,
            hrtf_profile: self.rain_view.hrtf_profile.clone(),
            webgpu_fp16_enabled: self.rain_view.webgpu_fp16,
            show_advanced_inspector: self.rain_view.show_advanced_inspector,
            show_spectrogram: self.rain_view.show_spectrogram,
            custom_ir_hash: self
                .rain_view
                .custom_ir_meta
                .as_ref()
                .map(|m| m.sha256_hash.clone()),
        };
        save_session_state(Some(storage), &session);
        mark_first_launch_done(Some(storage));

        // Native: the full state goes into eframe's storage file. The browser keeps it in
        // IndexedDB instead, so a local storage copy would only duplicate it.
        #[cfg(not(target_arch = "wasm32"))]
        {
            eframe::set_value(storage, eframe::APP_KEY, &self.state);
            if let Ok(json_str) = serde_json::to_string(&self.state) {
                storage.set_string(DEDICATED_STORAGE_KEY, json_str);
            }
        }
        storage.flush();
        self.persist_state();
    }

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.apply_storage_events();
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

        // 2. Render bottom spectrogram waterfall panel (Decoupled, resizable, default visible)
        if self.rain_view.show_spectrogram {
            egui::Panel::bottom("spectrogram_bottom_panel")
                .resizable(true)
                .default_size(120.0)
                .min_size(60.0)
                .max_size(220.0)
                .show(ui, |ui| {
                    render_spectrogram_panel(
                        ui,
                        &self.spectrogram_history,
                        &mut self.rain_view.show_spectrogram,
                    );
                });
        }

        // 3. Central content area
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.group(|ui| {
                    ui.heading("🌧 RainAI Neural Spatial Soundscape Studio");
                    ui.label("Continuous, non-repetitive procedural rain synthesis conditioned on 64 physical parameters.");
                    ui.add_space(8.0);
                    self.rain_view.render(ui, &mut self.state.rain, self.audio_state.as_ref());
                });
            });
        });

        // 4. Modals, warnings, and background task queue dock
        components::modals::render_dialogs(self, ui);
        components::modals::render_warning_banners(self, ui.ctx());
        components::modals::render_queue_dock(self, ui.ctx());
    }
}

#[cfg(target_os = "android")]
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use eframe::NativeOptions;
    let mut options = NativeOptions::default();
    options.android_app = Some(app);
    eframe::run_native(
        "RainAI",
        options,
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
    )
    .unwrap();
}

#[cfg(target_os = "ios")]
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn ios_main() {
    use eframe::NativeOptions;
    let _ = audio::IosAudioSessionManager::configure_audio_session();
    let options = NativeOptions::default();
    let _ = eframe::run_native(
        "RainAI",
        options,
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
    );
}
