import init, { WasmInferenceNode } from './pkg/web.js';

class RainInferenceProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.initialized = false;
        this.inferenceNode = null;
        this.conditioningBuffer = new Float32Array(554);
        this.sharedConditioning = null;
        
        this.port.onmessage = async (event) => {
            const { type, payload } = event.data;
            
            if (type === 'INIT_WASM') {
                try {
                    await init(payload.wasmModule);
                    this.inferenceNode = new WasmInferenceNode();
                    this.initialized = true;
                    this.port.postMessage({ type: 'READY' });
                } catch (error) {
                    this.port.postMessage({ type: 'ERROR', message: error.toString() });
                }
            } else if (type === 'SET_SHARED_BUFFER') {
                // Lock-free high-rate updates mapped directly to the UI thread
                this.sharedConditioning = new Float32Array(payload.sharedBuffer);
            } else if (type === 'SET_EXPERTS') {
                if (this.inferenceNode) this.inferenceNode.set_active_experts(payload.experts);
            }
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        
        if (!this.initialized || output.length < 4) {
            return true; 
        }

        const [channelW, channelX, channelY, channelZ] = output;
        const bufferSize = channelW.length;

        for (let i = 0; i < bufferSize; i++) {
            if (this.sharedConditioning) {
                this.conditioningBuffer.set(this.sharedConditioning);
            }

            try {
                // Execute the SIMD WASM pipeline for this frame
                const foaFrame = this.inferenceNode.step_frame(this.conditioningBuffer);
                
                channelW[i] = foaFrame[0];
                channelX[i] = foaFrame[1];
                channelY[i] = foaFrame[2];
                channelZ[i] = foaFrame[3];
            } catch (e) {
                channelW[i] = channelX[i] = channelY[i] = channelZ[i] = 0.0;
            }
        }

        return true;
    }
}

registerProcessor('rain-inference-processor', RainInferenceProcessor);
