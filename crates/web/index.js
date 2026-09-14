// ============================================================================
// WebAudio Autoplay Policy Unlocking (Safari / iOS & Mobile Chrome)
// ============================================================================
(function () {
  const unlockAudio = () => {
    if (window.__rainAudioContext && window.__rainAudioContext.state === 'suspended') {
      window.__rainAudioContext.resume().catch(() => {});
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
const IDB_DB_NAME = 'app_template_offline_store';
const IDB_STORE_NAME = 'app_template_kv';
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
