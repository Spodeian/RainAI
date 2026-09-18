# 🖥️ RainAI Native Desktop Studio (`desktop`)

The `desktop` crate provides the native desktop executable for Windows, macOS, and Linux.

## 🚀 Execution

```bash
# Run release build with mimalloc allocator
cargo run -p desktop --release
```

## ⚙️ Features
- Backed by the **`mimalloc`** high-performance memory allocator for zero-latency audio allocations.
- Multi-threaded rendering with `eframe` native viewport configuration.
- Native file dialogues for audio export and custom preset loading.
