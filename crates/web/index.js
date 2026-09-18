// ============================================================================
// RainAI Neural Audio Engine Initialization & Autoplay Unlocking
// ============================================================================
window.__rainAudioContext = new (window.AudioContext || window.webkitAudioContext)();
window.__rainEngine = null;

async function initializeAudioEngine() {
  if (window.__rainEngine) return window.__rainEngine;

  const audioCtx = window.__rainAudioContext;
  
  // 1. Load the AudioWorklet module
  await audioCtx.audioWorklet.addModule('inference_worklet.js');
  
  // 2. Fetch and compile the WebAssembly binary dynamically
  const response = await fetch('./pkg/web_bg.wasm');
  const wasmBytes = await response.arrayBuffer();
  const wasmModule = await WebAssembly.compile(wasmBytes);
  
  // 3. Create the Inference Node (0 inputs, 1 output with 4 channels)
  const inferenceNode = new AudioWorkletNode(audioCtx, 'rain-inference-processor', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [4]
  });
  
  // 4. Initialize lock-free memory mapping (554 f32s = 2216 bytes)
  const sharedBuffer = new SharedArrayBuffer(554 * 4);
  const telemetryArray = new Float32Array(sharedBuffer);
  
  // 5. Send initialization payloads to the Worklet
  inferenceNode.port.postMessage({ type: 'INIT_WASM', payload: { wasmModule } });
  inferenceNode.port.postMessage({ type: 'SET_SHARED_BUFFER', payload: { sharedBuffer } });
  
  // 6. Connect the 4-channel FOA output to the destination (or a Binaural downmixer)
  inferenceNode.connect(audioCtx.destination);
  
  window.__rainEngine = { audioCtx, inferenceNode, telemetryArray };
  console.log("RainAI Audio Engine Initialized");
  
  return window.__rainEngine;
}

(function () {
  const unlockAudio = async () => {
    if (window.__rainAudioContext && window.__rainAudioContext.state === 'suspended') {
      window.__rainAudioContext.resume().catch(() => {});
    }
    // Initialize the WASM engine on the first user interaction
    if (!window.__rainEngine) {
      await initializeAudioEngine().catch(e => console.error("Audio Engine Init Failed:", e));
    }
  };
  ['click', 'touchstart', 'pointerdown', 'keydown'].forEach((evt) => {
    window.addEventListener(evt, unlockAudio, { passive: true });
  });
})();

// ============================================================================
// Service Worker Registration for Offline PWA Capabilities
// ============================================================================
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('./sw.js')
      .then((reg) => console.log('ServiceWorker active on scope:', reg.scope))
      .catch((err) => console.log('ServiceWorker registration deferred:', err));
  });
}

// ============================================================================
// Progressive Web App (PWA) Install Prompt Engine
// ============================================================================
window.__pwaInstallPrompt = null;
window.__pwaInstallAvailable = false;
window.__pwaInstalled = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true;

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  window.__pwaInstallPrompt = e;
  window.__pwaInstallAvailable = true;
  window.dispatchEvent(new CustomEvent('pwa-install-available'));
  console.log('PWA installation prompt captured and ready');
});

window.addEventListener('appinstalled', () => {
  window.__pwaInstallPrompt = null;
  window.__pwaInstallAvailable = false;
  window.__pwaInstalled = true;
  window.dispatchEvent(new CustomEvent('pwa-installed'));
  console.log('PWA successfully installed by user');
  if (window.__requestPersistentStorage) {
    window.__requestPersistentStorage();
  }
});

window.__triggerPWAInstall = async function () {
  if (!window.__pwaInstallPrompt) {
    console.warn('PWA install prompt is not available at this time');
    return false;
  }
  try {
    window.__pwaInstallPrompt.prompt();
    const { outcome } = await window.__pwaInstallPrompt.userChoice;
    console.log(`User response to PWA install prompt: ${outcome}`);
    if (outcome === 'accepted') {
      window.__pwaInstallPrompt = null;
      window.__pwaInstallAvailable = false;
      return true;
    }
    return false;
  } catch (err) {
    console.error('Error triggering PWA install:', err);
    return false;
  }
};

// ============================================================================
// StorageManager Persistence API Bridge
// ============================================================================
window.__storagePersisted = false;

window.__checkStoragePersisted = async function () {
  if (navigator.storage && navigator.storage.persisted) {
    try {
      const persisted = await navigator.storage.persisted();
      window.__storagePersisted = persisted;
      return persisted;
    } catch (e) {
      console.warn('Failed to check storage persistence:', e);
      return false;
    }
  }
  return false;
};

window.__requestPersistentStorage = async function () {
  if (navigator.storage && navigator.storage.persist) {
    try {
      const granted = await navigator.storage.persist();
      window.__storagePersisted = granted;
      window.dispatchEvent(new CustomEvent('storage-persistence-changed', { detail: { granted } }));
      console.log(`Persistent storage requested. Granted: ${granted}`);
      return granted;
    } catch (e) {
      console.error('Error requesting persistent storage:', e);
      return false;
    }
  }
  return false;
};

window.__getStorageEstimate = async function () {
  if (navigator.storage && navigator.storage.estimate) {
    try {
      const estimate = await navigator.storage.estimate();
      return JSON.stringify({
        usage: estimate.usage || 0,
        quota: estimate.quota || 0,
      });
    } catch (e) {
      return JSON.stringify({ usage: 0, quota: 0 });
    }
  }
  return JSON.stringify({ usage: 0, quota: 0 });
};

// Auto-check persistence status on load
window.addEventListener('load', () => {
  if (window.__checkStoragePersisted) {
    window.__checkStoragePersisted();
  }
});

// ============================================================================
// Robust IndexedDB Fallback / Multi-Tier Storage Engine
// ============================================================================
const IDB_DB_NAME = 'rainai_offline_store';
const IDB_STORE_NAME = 'rainai_kv';
const IDB_VERSION = 1;

function openIdbDatabase() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is not supported in this environment'));
      return;
    }
    const request = window.indexedDB.open(IDB_DB_NAME, IDB_VERSION);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(IDB_STORE_NAME)) {
        db.createObjectStore(IDB_STORE_NAME);
      }
    };
    request.onsuccess = (e) => resolve(e.target.result);
    request.onerror = (e) => reject(e.target.error);
  });
}

window.__saveToIndexedDB = async function (key, value) {
  try {
    const db = await openIdbDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(IDB_STORE_NAME);
      const req = store.put(value, key);
      req.onsuccess = () => resolve(true);
      req.onerror = (e) => reject(e.target.error);
    });
  } catch (err) {
    console.error('IndexedDB save error for key:', key, err);
    return false;
  }
};

window.__loadFromIndexedDB = async function (key) {
  try {
    const db = await openIdbDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE_NAME, 'readonly');
      const store = tx.objectStore(IDB_STORE_NAME);
      const req = store.get(key);
      req.onsuccess = (e) => resolve(e.target.result || null);
      req.onerror = (e) => reject(e.target.error);
    });
  } catch (err) {
    console.error('IndexedDB load error for key:', key, err);
    return null;
  }
};

window.__deleteFromIndexedDB = async function (key) {
  try {
    const db = await openIdbDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(IDB_STORE_NAME);
      const req = store.delete(key);
      req.onsuccess = () => resolve(true);
      req.onerror = (e) => reject(e.target.error);
    });
  } catch (err) {
    return false;
  }
};

// ============================================================================
// Screen Wake Lock API (Prevents Mobile Sleep Throttling During Playback)
// ============================================================================
window.__wakeLockSentinel = null;
window.__wakeLockDesired = false;

window.__setWakeLock = async function (active) {
  window.__wakeLockDesired = active;
  if (!('wakeLock' in navigator)) {
    return false;
  }

  try {
    if (active) {
      if (!window.__wakeLockSentinel) {
        window.__wakeLockSentinel = await navigator.wakeLock.request('screen');
        window.__wakeLockSentinel.addEventListener('release', () => {
          window.__wakeLockSentinel = null;
          console.log('[RainAI] Screen Wake Lock released');
        });
        console.log('[RainAI] Screen Wake Lock acquired for continuous audio playback');
      }
      return true;
    } else {
      if (window.__wakeLockSentinel) {
        await window.__wakeLockSentinel.release();
        window.__wakeLockSentinel = null;
      }
      return true;
    }
  } catch (err) {
    console.warn('[RainAI] Screen Wake Lock error:', err);
    return false;
  }
};

// Re-acquire Wake Lock when tab becomes visible if playback was active
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && window.__wakeLockDesired) {
    await window.__setWakeLock(true);
  }
});

