#!/usr/bin/env bash
# ==============================================================================
# RainAI Unified Build & Deployment Pipeline
# Coordinates Python ML exports, Rust WebAssembly builds, and Cloudflare Pages
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$ROOT_DIR/crates/web/dist"

echo "======================================================================"
echo "  RainAI Unified Deployment Pipeline"
echo "======================================================================"

# ==========================================
# PHASE 1: AI Model Compilation & Export
# ==========================================
echo -e "\n--- Stage 1: Compiling Multi-Backend Model Artifacts ---"
cd "$ROOT_DIR"

PYTHON_EXE="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_EXE=".venv/Scripts/python.exe"
elif command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
fi

echo "[*] Using Python environment: $($PYTHON_EXE --version)"
"$PYTHON_EXE" -X utf8 src/export/export_all.py

# Workspace-anchored export target unified with crates/inference/data
EXPORT_DIR="$ROOT_DIR/crates/inference/data"
SLICES_DIR="$EXPORT_DIR/wasm"
DEPLOY_CONFIG="$EXPORT_DIR/rainai_deployment_config.json"

if [ ! -f "$DEPLOY_CONFIG" ]; then
    echo "[!] Error: $DEPLOY_CONFIG was not generated!"
    exit 1
fi

# ==========================================
# PHASE 2: UI WebAssembly Build
# ==========================================
echo -e "\n--- Stage 2: Compiling Rust Web Application ---"
export NODE_ENV="production"
export NODE_VERSION="${NODE_VERSION:-24}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PATH="$CARGO_HOME/bin:$PATH"

# Toolchain check
RUST_TOOLCHAIN="stable"
if [ -f "rust-toolchain.toml" ]; then
    RUST_TOOLCHAIN=$(grep -E '^\s*channel\s*=' rust-toolchain.toml | head -n 1 | cut -d '"' -f 2 | tr -d ' ' || true)
fi

if ! command -v rustup &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "$RUST_TOOLCHAIN" --target wasm32-unknown-unknown
else
    rustup target add wasm32-unknown-unknown 2>/dev/null || true
fi

# Trunk Installation
if ! command -v trunk &> /dev/null; then
    echo "Downloading and caching latest Trunk asset bundler..."
    wget -qO- https://github.com/trunk-rs/trunk/releases/latest/download/trunk-x86_64-unknown-linux-gnu.tar.gz | tar -xzf - -C "$CARGO_HOME/bin"
fi
TRUNK_BIN=$(command -v trunk || echo "$CARGO_HOME/bin/trunk")

echo "Purging previous build distribution caches..."
rm -rf "$DIST_DIR" dist

export RUSTFLAGS="-C target-feature=+simd128,+bulk-memory,+mutable-globals,+nontrapping-fptoint,+sign-ext,+reference-types,+multivalue -C link-arg=-zstack-size=2097152 ${RUSTFLAGS:-}"

"$TRUNK_BIN" build crates/web/index.html --release --public-url "/"

# ==========================================
# PHASE 3: Asset Synchronization
# ==========================================
echo -e "\n--- Stage 3: Synchronizing AI Components to Frontend Runtimes ---"
INFERENCE_DATA="$EXPORT_DIR"
WEB_MODELS="$DIST_DIR/models"
WEB_SHADERS="$DIST_DIR/shaders"
WEB_DATA="$DIST_DIR/data"

mkdir -p "$INFERENCE_DATA" "$WEB_MODELS" "$WEB_SHADERS" "$WEB_DATA"

# Sync deployment config to local inference runtime
cp -f "$DEPLOY_CONFIG" "$INFERENCE_DATA/rainai_deployment_config.json"

# Sync progressive slices and deployment config to Web distribution
cp -f "$SLICES_DIR"/*.bin "$WEB_MODELS/" 2>/dev/null || true
cp -f "$DEPLOY_CONFIG" "$WEB_DATA/rainai_deployment_config.json"

# Sync WebGPU/DSP shaders if present
if [ -d "$ROOT_DIR/crates/inference/src/shaders" ]; then
    cp -f "$ROOT_DIR"/crates/inference/src/shaders/*.wgsl "$WEB_SHADERS/" 2>/dev/null || true
elif [ -d "$ROOT_DIR/src/dsp/shaders" ]; then
    cp -f "$ROOT_DIR/src/dsp/shaders/"*.wgsl "$WEB_SHADERS/" 2>/dev/null || true
fi

# ==========================================
# PHASE 4: Optimization & Compression
# ==========================================
echo -e "\n--- Stage 4: Production Asset Optimization ---"
WASM_OPT_BIN="wasm-opt"
if ! command -v wasm-opt &> /dev/null; then
    BINARYEN_VERSION="version_132"
    wget -qO /tmp/binaryen.tar.gz "https://github.com/WebAssembly/binaryen/releases/download/${BINARYEN_VERSION}/binaryen-${BINARYEN_VERSION}-x86_64-linux.tar.gz"
    tar -xzf /tmp/binaryen.tar.gz -C /tmp
    find /tmp -name "wasm-opt" -type f -exec mv {} "$CARGO_HOME/bin/wasm-opt" \;
    WASM_OPT_BIN="$CARGO_HOME/bin/wasm-opt"
fi

for wasm_file in "$DIST_DIR"/*.wasm; do
    if [ -f "$wasm_file" ]; then
        "$WASM_OPT_BIN" -Oz --enable-simd --enable-bulk-memory --enable-reference-types "$wasm_file" -o "$wasm_file" || true
    fi
done

BUILD_ID=$(git rev-parse --short HEAD 2>/dev/null || date +%s)
if [ -f "$DIST_DIR/sw.js" ]; then
    sed -i "s/CACHE_NAME = '.*'/CACHE_NAME = 'serverless-desktop-template-cache-${BUILD_ID}'/g" "$DIST_DIR/sw.js" 2>/dev/null || true
fi

if command -v npx &> /dev/null; then
    for js_file in "$DIST_DIR"/*.js; do npx --yes esbuild "$js_file" --minify --allow-overwrite --outfile="$js_file" || true; done
    for css_file in "$DIST_DIR"/*.css; do npx --yes esbuild "$css_file" --minify --allow-overwrite --outfile="$css_file" || true; done
fi

if command -v brotli &> /dev/null; then
    find "$DIST_DIR" -type f \( -name "*.wasm" -o -name "*.js" -o -name "*.css" -o -name "*.html" -o -name "*.json" -o -name "*.bin" \) -exec brotli -f -k -q 11 {} + 2>/dev/null || true
fi

cp -f crates/web/_headers "$DIST_DIR/_headers" 2>/dev/null || true
cp -f crates/web/_redirects "$DIST_DIR/_redirects" 2>/dev/null || true

# ==========================================
# PHASE 5: Deployment & Benchmarks
# ==========================================
echo -e "\n--- Stage 5: Cloudflare Deployment ---"
if [ "${CLOUDFLARE_WORKER_DEPLOY:-false}" = "true" ]; then
    wrangler deploy
else
    echo "Pages / Static CDN deployment context detected. Build ready for publishing."
fi

if [ "${RUN_BENCHMARKS:-false}" = "true" ]; then
    echo -e "\n--- Stage 6: Benchmarking Multi-Backend Inference Performance ---"
    "$PYTHON_EXE" -X utf8 src/benchmarks/benchmark_backends.py --iterations 50
fi

echo -e "\n[+] Deployment Build Completed Successfully!"
