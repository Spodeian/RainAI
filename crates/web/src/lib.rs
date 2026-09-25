//! WebAssembly (WASM) Entry Point and PWA Runner for RainAI.
//!
//! Handles client-side WebGPU shader pipeline initialization, WebAudio audio worklet contexts,
//! canvas resizing, console panic hooks, and JavaScript/TypeScript interop bindings.

#[cfg(target_arch = "wasm32")]
use app::TemplateApp;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, prelude::*};

#[cfg(target_arch = "wasm32")]
use inference::{runner::InferenceRunner, weight_loader::WeightLoader};
#[cfg(target_arch = "wasm32")]
use shared::rain::{QualityTier, CONDITION_DIM};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn main() {
    // Universal telemetry & logging initialization
    spodeian_telemetry::init_default();

    // Spawn the async eframe WebRunner natively
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .and_then(|win| win.document())
            .expect("Failed to get document");
        let canvas = document
            .get_element_by_id("egui_canvas")
            .expect("Canvas element 'egui_canvas' not found")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Failed to cast element to HtmlCanvasElement");

        let web_options = eframe::WebOptions::default();
        let runner = eframe::WebRunner::new();
        let _ = runner
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
            )
            .await;
    });
}

/// WASM binding for the AudioWorklet to stream conditioning vectors into the SIMD pipeline.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmInferenceNode {
    runner: InferenceRunner,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmInferenceNode {
    /// Initializes the inference engine and loads the embedded ternary weights.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmInferenceNode, JsValue> {
        let cache = WeightLoader::load_embedded_ternary()
            .map_err(|e| JsValue::from_str(&format!("Failed to load embedded weights: {}", e)))?;
            
        let runner = InferenceRunner::new(QualityTier::Ternary158, cache);
        Ok(Self { runner })
    }

    /// Processes a single frame of conditioning data and returns 4-channel FOA (W, X, Y, Z).
    pub fn step_frame(&mut self, conditioning: &[f32]) -> Result<Vec<f32>, JsValue> {
        if conditioning.len() != CONDITION_DIM {
            return Err(JsValue::from_str(&format!(
                "Conditioning vector must be exactly {} elements",
                CONDITION_DIM
            )));
        }
        
        let mut cond_array = [0.0f32; CONDITION_DIM];
        cond_array.copy_from_slice(conditioning);
        
        let (w, x, y, z) = self.runner.step(&cond_array);
        
        // Return as a Vec<f32>, which wasm_bindgen translates to a Float32Array
        Ok(vec![w, x, y, z])
    }
    
    /// Dynamically adjust the active MoE experts to throttle CPU usage.
    pub fn set_active_experts(&mut self, experts: usize) {
        self.runner.set_active_experts(experts);
    }
}
