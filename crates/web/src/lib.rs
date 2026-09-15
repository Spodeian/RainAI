//! Web entrypoint and WebRunner for the Serverless & Desktop Template.

#[cfg(target_arch = "wasm32")]
use app::TemplateApp;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, prelude::*};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn main() {
    // Redirect panic logs to browser developer console
    console_error_panic_hook::set_once();
    tracing_subscriber::fmt()
        .with_writer(tracing_web::MakeConsoleWriter)
        .without_time()
        .with_ansi(false)
        .init();

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
