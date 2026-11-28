//! Unified multi-tiered storage engine, persistence manager, PWA install bridge, and diagnostics for template app.

use audio::decoder::DecodeMode;
use inference::compute_router::FlowSolverAlgorithm;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use spodeian_cache::StorageTier;
#[allow(unused_imports)]
use spodeian_cache::{ContentAddressedStorage, PreferentialRouter};
#[allow(unused_imports)]
use tracing::{error, info, warn};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StorageBackend {
    #[default]
    LocalStorage,
    IndexedDb,
    CacheApi,
    NativeCas,
    MemoryOnly,
}

impl StorageBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::LocalStorage => "Local Storage (Fast Tier)",
            Self::IndexedDb => "IndexedDB (Structured Relational Tier)",
            Self::CacheApi => "Cache API (Large Binary Weights Tier)",
            Self::NativeCas => "Native CAS (Content-Addressed Disk Tier)",
            Self::MemoryOnly => "In-Memory Only (Ephemeral)",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StorageDiagnostics {
    pub is_persisted: Option<bool>,
    pub pwa_install_available: bool,
    pub is_pwa_installed: bool,
    pub backend: StorageBackend,
    pub quota_exceeded: bool,
    pub idb_active: bool,
    pub usage_bytes: u64,
    pub quota_bytes: u64,
}

/// Query current storage persistence and PWA status from browser environment
#[allow(unused_mut)]
pub fn query_storage_diagnostics() -> StorageDiagnostics {
    let mut diag = StorageDiagnostics::default();

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            // Check if PWA is installed or installable
            if let Ok(val) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__pwaInstallAvailable"),
            ) {
                diag.pwa_install_available = val.as_bool().unwrap_or(false);
            }
            if let Ok(val) =
                js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__pwaInstalled"))
            {
                diag.is_pwa_installed = val.as_bool().unwrap_or(false);
            }

            // Check persistence state
            if let Ok(val) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__storagePersisted"),
            ) {
                if let Some(b) = val.as_bool() {
                    diag.is_persisted = Some(b);
                }
            }
        }
    }

    diag
}

/// Request persistent storage from the browser (immune to automatic eviction)
pub fn request_persistent_storage() {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(func) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__requestPersistentStorage"),
            ) {
                if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                    let _ = func.call0(&window);
                    info!("Triggered __requestPersistentStorage from template UI");
                }
            }
        }
    }
}

/// Trigger the native PWA installation prompt
pub fn trigger_pwa_install() {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(func) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__triggerPWAInstall"),
            ) {
                if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                    let _ = func.call0(&window);
                    info!("Triggered __triggerPWAInstall from template UI");
                }
            }
        }
    }
}

pub const DEDICATED_STORAGE_KEY: &str = "serverless_template_app_state";
pub const FIRST_LAUNCH_STORAGE_KEY: &str = "rainai_first_launch_done";
pub const SESSION_STORAGE_KEY: &str = "rainai_persistent_session_state";

fn default_true() -> bool {
    true
}

/// Persistent session parameters surviving reboots and browser relaunches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistentSessionState {
    pub flow_solver: FlowSolverAlgorithm,
    pub active_preset_name: String,
    pub master_volume: f32,
    pub decode_mode: DecodeMode,
    pub noise_masking_enabled: bool,
    pub noise_masking_threshold_db: f32,
    pub hrtf_profile: String,
    pub webgpu_fp16_enabled: bool,
    pub show_advanced_inspector: bool,
    #[serde(default = "default_true")]
    pub show_spectrogram: bool,
    #[serde(default)]
    pub custom_ir_hash: Option<String>,
}

impl Default for PersistentSessionState {
    fn default() -> Self {
        Self {
            flow_solver: FlowSolverAlgorithm::AdaptiveRk45 {
                tol: 1e-3,
                initial_h: 0.1,
            },
            active_preset_name: "Gentle Summer Rain".to_string(),
            master_volume: 0.60,
            decode_mode: DecodeMode::StereoSpeakers, // Desktop speakers default
            noise_masking_enabled: false,
            noise_masking_threshold_db: -40.0,
            hrtf_profile: "Kemar-Compact-Standard".to_string(),
            webgpu_fp16_enabled: true,
            show_advanced_inspector: false,
            show_spectrogram: true,
            custom_ir_hash: None,
        }
    }
}

pub const SETTINGS_STORAGE_KEY: &str = "rainai_settings_state_v1";

pub fn load_settings_state(
    storage: Option<&dyn eframe::Storage>,
) -> crate::settings::SettingsState {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(local_storage)) = window.local_storage() {
                if let Ok(Some(content)) = local_storage.get_item(SETTINGS_STORAGE_KEY) {
                    if let Ok(settings) =
                        serde_json::from_str::<crate::settings::SettingsState>(&content)
                    {
                        return settings;
                    }
                }
            }
        }
    }

    if let Some(storage) = storage {
        if let Some(raw) = storage.get_string(SETTINGS_STORAGE_KEY) {
            if let Ok(settings) = serde_json::from_str::<crate::settings::SettingsState>(&raw) {
                return settings;
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = std::env::temp_dir().join(SETTINGS_STORAGE_KEY);
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(settings) = serde_json::from_str::<crate::settings::SettingsState>(&content) {
                return settings;
            }
        }
    }

    // Migration fallback: check older PersistentSessionState
    let session = load_session_state(storage);
    crate::settings::SettingsState {
        master_volume: session.master_volume,
        noise_masking_enabled: session.noise_masking_enabled,
        noise_masking_threshold_db: session.noise_masking_threshold_db,
        hrtf_profile: session.hrtf_profile,
        custom_ir_hash: session.custom_ir_hash,
        show_spectrogram: session.show_spectrogram,
        show_advanced_settings: session.show_advanced_inspector,
        active_preset_name: session.active_preset_name,
        ..Default::default()
    }
}

pub fn save_settings_state(
    storage: Option<&mut dyn eframe::Storage>,
    settings: &crate::settings::SettingsState,
) {
    if let Ok(json_str) = serde_json::to_string(settings) {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(Some(local_storage)) = window.local_storage() {
                    let _ = local_storage.set_item(SETTINGS_STORAGE_KEY, &json_str);
                }
            }
        }

        if let Some(storage) = storage {
            storage.set_string(SETTINGS_STORAGE_KEY, json_str.clone());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = std::env::temp_dir().join(SETTINGS_STORAGE_KEY);
            let _ = std::fs::write(path, json_str);
        }
    }
}

pub fn is_first_launch(storage: Option<&dyn eframe::Storage>) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(local_storage)) = window.local_storage() {
                if let Ok(Some(val)) = local_storage.get_item(FIRST_LAUNCH_STORAGE_KEY) {
                    return val != "true";
                }
            }
        }
    }

    if let Some(storage) = storage {
        if let Some(val) = storage.get_string(FIRST_LAUNCH_STORAGE_KEY) {
            return val != "true";
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let flag_path = std::env::temp_dir().join(FIRST_LAUNCH_STORAGE_KEY);
        if flag_path.exists() {
            return false;
        }
    }

    true
}

pub fn mark_first_launch_done(storage: Option<&mut dyn eframe::Storage>) {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(local_storage)) = window.local_storage() {
                let _ = local_storage.set_item(FIRST_LAUNCH_STORAGE_KEY, "true");
            }
        }
    }

    if let Some(storage) = storage {
        storage.set_string(FIRST_LAUNCH_STORAGE_KEY, "true".to_string());
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let flag_path = std::env::temp_dir().join(FIRST_LAUNCH_STORAGE_KEY);
        let _ = std::fs::write(flag_path, "true");
    }
}

pub fn load_session_state(storage: Option<&dyn eframe::Storage>) -> PersistentSessionState {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(local_storage)) = window.local_storage() {
                if let Ok(Some(content)) = local_storage.get_item(SESSION_STORAGE_KEY) {
                    if let Ok(session) = serde_json::from_str::<PersistentSessionState>(&content) {
                        return session;
                    }
                }
            }
        }
    }

    if let Some(storage) = storage {
        if let Some(raw) = storage.get_string(SESSION_STORAGE_KEY) {
            if let Ok(session) = serde_json::from_str::<PersistentSessionState>(&raw) {
                return session;
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = std::env::temp_dir().join(SESSION_STORAGE_KEY);
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(session) = serde_json::from_str::<PersistentSessionState>(&content) {
                return session;
            }
        }
    }

    PersistentSessionState::default()
}

pub fn save_session_state(
    storage: Option<&mut dyn eframe::Storage>,
    session: &PersistentSessionState,
) {
    if let Ok(json_str) = serde_json::to_string(session) {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(Some(local_storage)) = window.local_storage() {
                    let _ = local_storage.set_item(SESSION_STORAGE_KEY, &json_str);
                }
            }
        }

        if let Some(storage) = storage {
            storage.set_string(SESSION_STORAGE_KEY, json_str.clone());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = std::env::temp_dir().join(SESSION_STORAGE_KEY);
            let _ = std::fs::write(path, json_str);
        }
    }
}

/// Robust dual-format deserializer for AppState, attempting JSON first and falling back to RON.
pub fn deserialize_app_state(content: &str) -> Result<shared::AppState, String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("Storage content is empty".to_string());
    }

    // 1. Attempt JSON deserialization
    match serde_json::from_str::<shared::AppState>(trimmed) {
        Ok(state) => Ok(state),
        Err(json_err) => {
            // 2. Attempt RON deserialization
            match ron::from_str::<shared::AppState>(trimmed) {
                Ok(state) => Ok(state),
                Err(ron_err) => Err(format!(
                    "Failed to deserialize AppState: JSON error: {}; RON error: {}",
                    json_err, ron_err
                )),
            }
        }
    }
}

/// Multi-tiered loader for AppState.
/// Checks window.localStorage (on wasm32) and eframe::Storage across both dedicated and legacy keys,
/// supporting both JSON and RON formats seamlessly.
pub fn load_state_multi_tier(storage: Option<&dyn eframe::Storage>) -> Option<shared::AppState> {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(local_storage)) = window.local_storage() {
                // Tier 1: Check dedicated key in browser localStorage
                if let Ok(Some(content)) = local_storage.get_item(DEDICATED_STORAGE_KEY) {
                    match deserialize_app_state(&content) {
                        Ok(state) => {
                            info!(
                                "Successfully restored AppState from localStorage [{}]",
                                DEDICATED_STORAGE_KEY
                            );
                            return Some(state);
                        }
                        Err(e) => {
                            warn!(
                                "Failed to parse AppState from localStorage [{}]: {}",
                                DEDICATED_STORAGE_KEY, e
                            );
                        }
                    }
                }

                // Tier 2: Check standard 'app' key in browser localStorage (fallback/legacy)
                if let Ok(Some(content)) = local_storage.get_item(eframe::APP_KEY) {
                    match deserialize_app_state(&content) {
                        Ok(state) => {
                            info!(
                                "Successfully restored AppState from localStorage [{}]",
                                eframe::APP_KEY
                            );
                            return Some(state);
                        }
                        Err(e) => {
                            warn!(
                                "Failed to parse AppState from localStorage [{}]: {}",
                                eframe::APP_KEY,
                                e
                            );
                        }
                    }
                }
            }
        }
    }

    // Tier 3: Check eframe::Storage
    if let Some(storage) = storage {
        // Check dedicated key in eframe storage
        if let Some(raw) = storage.get_string(DEDICATED_STORAGE_KEY) {
            match deserialize_app_state(&raw) {
                Ok(state) => {
                    info!(
                        "Successfully restored AppState from eframe::Storage [{}]",
                        DEDICATED_STORAGE_KEY
                    );
                    return Some(state);
                }
                Err(e) => {
                    warn!(
                        "Failed to parse AppState from eframe::Storage [{}]: {}",
                        DEDICATED_STORAGE_KEY, e
                    );
                }
            }
        }

        // Check 'app' key string in eframe storage
        if let Some(raw) = storage.get_string(eframe::APP_KEY) {
            match deserialize_app_state(&raw) {
                Ok(state) => {
                    info!(
                        "Successfully restored AppState from eframe::Storage [{}]",
                        eframe::APP_KEY
                    );
                    return Some(state);
                }
                Err(e) => {
                    warn!(
                        "Failed to parse AppState from eframe::Storage [{}]: {}",
                        eframe::APP_KEY,
                        e
                    );
                }
            }
        }

        // Check native eframe::get_value (RON deserializer)
        if let Some(state) = eframe::get_value::<shared::AppState>(storage, eframe::APP_KEY) {
            info!("Successfully restored AppState from eframe::get_value (RON).");
            return Some(state);
        }
    }

    None
}

/// Save state using preferential multi-tiered routing:
/// - Cache API for large binaries / models (> 512 KB)
/// - IndexedDB for structured relational states
/// - LocalStorage for lightweight config (< 16 KB)
/// - ContentAddressedStorage on native desktop & mobile
pub fn save_state_multi_tier(key: &str, json_str: &str) -> Result<StorageBackend, String> {
    let size = json_str.len();
    let is_large_or_binary = size > 512 * 1024;
    let recommended_tier =
        PreferentialRouter::determine_tier(size, "application/json", is_large_or_binary);
    let _ = (key, &recommended_tier);

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            // Tier 1 (Large/Binary): If Cache API is recommended or payload is large
            if recommended_tier == StorageTier::CacheApi {
                if let Ok(func) = js_sys::Reflect::get(
                    &window,
                    &wasm_bindgen::JsValue::from_str("__saveToCacheApi"),
                ) {
                    if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                        let k = wasm_bindgen::JsValue::from_str(key);
                        let v = wasm_bindgen::JsValue::from_str(json_str);
                        let _ = func.call2(&window, &k, &v);
                        info!("Preferentially stored large asset to Cache API [{}]", key);
                        return Ok(StorageBackend::CacheApi);
                    }
                }
            }

            // Tier 2: Try localStorage for fast session data if small
            if size < 16 * 1024 {
                if let Ok(Some(storage)) = window.local_storage() {
                    if storage.set_item(key, json_str).is_ok() {
                        return Ok(StorageBackend::LocalStorage);
                    }
                }
            }

            // Tier 3: IndexedDB for structured entities and fallback
            if let Ok(func) = js_sys::Reflect::get(
                &window,
                &wasm_bindgen::JsValue::from_str("__saveToIndexedDB"),
            ) {
                if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                    let k = wasm_bindgen::JsValue::from_str(key);
                    let v = wasm_bindgen::JsValue::from_str(json_str);
                    let _ = func.call2(&window, &k, &v);
                    info!("Saved structured state to IndexedDB [{}]", key);
                    return Ok(StorageBackend::IndexedDb);
                }
            }

            return Err("All browser storage tiers failed.".to_string());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let cas_dir = std::env::temp_dir().join("rainai_cas_store");
        // One named entry per key, with superseded copies swept, so repeated saves replace
        // the state instead of adding a file each time.
        if let Ok(cas) = ContentAddressedStorage::new(&cas_dir)
            && cas.put_named(key, json_str.as_bytes()).is_ok()
        {
            if let Err(e) = cas.collect_garbage() {
                tracing::warn!(
                    "Could not remove superseded saves in {}: {e}",
                    cas_dir.display()
                );
            }
            return Ok(StorageBackend::NativeCas);
        }
    }

    Ok(StorageBackend::MemoryOnly)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_desktop_export_path(filename: &str) -> std::path::PathBuf {
    if let Ok(profile) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        let downloads = std::path::Path::new(&profile).join("Downloads");
        if downloads.is_dir() {
            return downloads.join(filename);
        }
    }
    std::path::PathBuf::from(filename)
}

/// Trigger client-side text file download via Blob URL
pub fn trigger_text_download(filename: &str, content: &str, mime_type: &str) {
    #[cfg(target_arch = "wasm32")]
    spodeian_web_utils::trigger_text_download(filename, content, mime_type);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime_type;
        let path = resolve_desktop_export_path(filename);
        match std::fs::write(&path, content) {
            Ok(()) => info!("Successfully wrote local file: {}", path.display()),
            Err(e) => error!("Failed to write export file '{}': {}", path.display(), e),
        }
    }
}

/// Trigger client-side binary file download (e.g. Compressed BSON) via Blob URL
pub fn trigger_binary_download(filename: &str, bytes: &[u8], mime_type: &str) {
    #[cfg(target_arch = "wasm32")]
    spodeian_web_utils::trigger_binary_download(filename, bytes, mime_type);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime_type;
        let path = resolve_desktop_export_path(filename);
        match std::fs::write(&path, bytes) {
            Ok(()) => info!("Successfully exported binary file: {}", path.display()),
            Err(e) => error!(
                "Failed to write binary export file '{}': {}",
                path.display(),
                e
            ),
        }
    }
}

/// Control browser Screen Wake Lock API to prevent mobile sleep during playback
pub fn set_screen_wake_lock(active: bool) {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(func) =
                js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__setWakeLock"))
            {
                if let Some(func) = func.dyn_ref::<js_sys::Function>() {
                    let arg = wasm_bindgen::JsValue::from_bool(active);
                    let _ = func.call1(&window, &arg);
                }
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = active;
    }
}

// ---------------------------------------------------------------------------
// Browser persistence: the whole state is one IndexedDB record, loaded once at startup
// and saved in the background (`spodeian_persist::WebStateStore`). Earlier versions
// saved large states only to IndexedDB but never read them back, so a reload restored an
// older local storage copy.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_STORE: std::cell::RefCell<Option<spodeian_persist::WebStateStore>> =
        const { std::cell::RefCell::new(None) };
}

/// Starts loading the saved state from IndexedDB (browser only). The result arrives
/// through [`take_loaded_web_state`]; `ctx` is repainted when it does.
pub fn open_web_store(ctx: &eframe::egui::Context) {
    #[cfg(target_arch = "wasm32")]
    {
        let ctx = ctx.clone();
        let legacy = spodeian_persist::LegacyLocations {
            local_storage_keys: vec![DEDICATED_STORAGE_KEY.into(), eframe::APP_KEY.into()],
            // The old JS database (`rainai_offline_store`) also caches model weights, so it stays.
            databases: vec![],
        };
        let store = spodeian_persist::WebStateStore::open(
            "rainai-state",
            DEDICATED_STORAGE_KEY,
            legacy,
            move || ctx.request_repaint(),
        );
        WEB_STORE.with(|s| *s.borrow_mut() = Some(store));
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = ctx;
}

/// The state loaded from IndexedDB, once: `Some(None)` means nothing was saved there yet
/// (save the current state to move it in). Always `None` on native.
pub fn take_loaded_web_state() -> Option<Option<shared::AppState>> {
    #[cfg(target_arch = "wasm32")]
    return WEB_STORE
        .with(|s| s.borrow().as_ref().and_then(|store| store.take_loaded()))
        .map(|saved| {
            saved.and_then(|json| {
                deserialize_app_state(&json)
                    .map_err(|e| tracing::warn!("Ignoring unreadable saved state: {e}"))
                    .ok()
            })
        });
    #[cfg(not(target_arch = "wasm32"))]
    None
}

/// The result of the most recent finished browser save, once: `Ok(())` when IndexedDB
/// confirmed it. Always `None` on native.
pub fn take_web_save_outcome() -> Option<Result<(), String>> {
    #[cfg(target_arch = "wasm32")]
    return WEB_STORE
        .with(|s| s.borrow().as_ref().and_then(|store| store.take_outcome()))
        .map(|outcome| match outcome {
            spodeian_persist::SaveOutcome::Saved => Ok(()),
            spodeian_persist::SaveOutcome::Failed(message) => Err(message),
            other => Err(format!("{other:?}")),
        });
    #[cfg(not(target_arch = "wasm32"))]
    None
}

/// Queues `json` to be written to IndexedDB. Returns `false` (doing nothing) on native or
/// before [`open_web_store`] ran, so the caller can fall back to its native save.
pub fn queue_web_save(json: &str) -> bool {
    #[cfg(target_arch = "wasm32")]
    return WEB_STORE.with(|s| {
        s.borrow().as_ref().is_some_and(|store| {
            store.queue_save(json.to_string());
            true
        })
    });
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = json;
        false
    }
}
