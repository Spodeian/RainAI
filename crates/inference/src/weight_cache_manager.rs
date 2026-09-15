//! Asynchronous weight streaming, IndexedDB browser caching, and dynamic tier hot-swapping.

use crate::weight_loader::{WeightCache, WeightLoader};
use crate::model::QuantizedModelManifest;
use shared::rain::QualityTier;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

/// Manages asynchronous fetching, local caching, and decoding of multi-tier model weights
pub struct WeightCacheManager;

impl WeightCacheManager {
    /// Asynchronously loads a weight tier, checking IndexedDB / local cache first before fetching over network.
    pub async fn load_tier(tier: QualityTier) -> Result<WeightCache, String> {
        let manifest = QuantizedModelManifest::load_default_ternary()
            .map_err(|e| format!("Failed to parse model manifest: {e}"))?;

        let cache_key = format!("rainai_weight_slice_{:?}", tier);

        // 1. Attempt to load from persistent local cache (IndexedDB on WASM, filesystem on Native)
        if let Some(cached_bytes) = Self::load_from_local_cache(&cache_key).await {
            tracing::info!("Restored weight tier {:?} from local cache storage", tier);
            return WeightLoader::load_from_manifest(&manifest, &cached_bytes, tier);
        }

        // 2. Fetch over network if not cached (WASM fetch or native client request)
        let slice_url = match tier {
            QualityTier::Ternary158 => "./data/slice_0_ternary.bin",
            QualityTier::AdaptiveMinimum => "./data/slice_1_adaptive.bin",
            QualityTier::HighInt16 => "./data/slice_2_int16.bin",
            QualityTier::StudioFp32 => "./data/slice_3_fp32.bin",
        };

        tracing::info!("Fetching weight tier {:?} from endpoint: {}", tier, slice_url);
        let raw_bytes = Self::fetch_binary_slice(slice_url).await?;

        // 3. Save to persistent local cache for future sessions
        Self::save_to_local_cache(&cache_key, &raw_bytes).await;

        // 4. Parse and unpack into WeightCache
        WeightLoader::load_from_manifest(&manifest, &raw_bytes, tier)
    }

    /// Retrieves binary slice from IndexedDB (WASM) or disk cache (Native)
    async fn load_from_local_cache(key: &str) -> Option<Vec<u8>> {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(func) = js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__loadFromIndexedDB")) {
                    if let Some(js_func) = func.dyn_ref::<js_sys::Function>() {
                        let promise = js_func.call1(&window, &wasm_bindgen::JsValue::from_str(key)).ok()?;
                        let result = JsFuture::from(js_sys::Promise::from(promise)).await.ok()?;
                        
                        // Convert JS ArrayBuffer / Uint8Array back to Rust Vec<u8>
                        if result.is_object() {
                            let uint8_arr = js_sys::Uint8Array::new(&result);
                            let mut vec = vec![0; uint8_arr.length() as usize];
                            uint8_arr.copy_to(&mut vec);
                            return Some(vec);
                        }
                    }
                }
            }
            None
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = std::env::temp_dir().join(format!("{}.bin", key));
            std::fs::read(path).ok()
        }
    }

    /// Persists binary slice into IndexedDB (WASM) or disk cache (Native)
    async fn save_to_local_cache(key: &str, bytes: &[u8]) {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(func) = js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__saveToIndexedDB")) {
                    if let Some(js_func) = func.dyn_ref::<js_sys::Function>() {
                        let k = wasm_bindgen::JsValue::from_str(key);
                        let uint8_arr = js_sys::Uint8Array::from(bytes);
                        let _ = js_func.call2(&window, &k, &uint8_arr.buffer());
                        tracing::info!("Persisted weight tier bytes to browser IndexedDB");
                    }
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = std::env::temp_dir().join(format!("{}.bin", key));
            let _ = std::fs::write(path, bytes);
        }
    }

    /// Helper to fetch binary data over HTTP in WASM or filesystem on native
    async fn fetch_binary_slice(url: &str) -> Result<Vec<u8>, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let window = web_sys::window().ok_or("No global window object found")?;
            let resp_value = JsFuture::from(window.fetch_with_str(url))
                .await
                .map_err(|e| format!("Network fetch failed: {e:?}"))?;
            
            let resp: web_sys::Response = resp_value.dyn_into()
                .map_err(|_| "Failed to cast fetch response")?;
            
            if !resp.ok() {
                return Err(format!("Server returned HTTP status error: {}", resp.status()));
            }

            let array_buffer_promise = resp.array_buffer()
                .map_err(|e| format!("Failed to get array buffer: {e:?}"))?;
            let array_buffer = JsFuture::from(array_buffer_promise)
                .await
                .map_err(|e| format!("Failed to await array buffer: {e:?}"))?;
            
            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            let mut bytes = vec![0; uint8_array.length() as usize];
            uint8_array.copy_to(&mut bytes);
            Ok(bytes)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::fs::read(url).map_err(|e| format!("Failed to read local file {url}: {e}"))
        }
    }
}
