// Service Worker for Serverless & Desktop Template
// Strategy: Cache-First with Background Network Revalidation (Stale-While-Revalidate)
const CACHE_NAME = 'serverless-desktop-template-cache-v20260912';

// Static assets to pre-cache on install
const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/favicon.svg',
  '/favicon.ico'
];

// 1. Pre-cache on install with error resilience and activate immediately
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return Promise.allSettled(
        PRECACHE_ASSETS.map((asset) =>
          fetch(asset, { cache: 'no-cache' }).then((response) => {
            if (response && response.ok) {
              return cache.put(asset, response);
            }
          }).catch(() => {})
        )
      );
    }).then(() => self.skipWaiting())
  );
});

// 2. Purge all legacy caches on activation and claim clients
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

// 3. Fetch router: Cache-First with Background Network Revalidation
self.addEventListener('fetch', (event) => {
  // Only handle local same-origin GET requests
  if (event.request.method !== 'GET' || !event.request.url.startsWith(self.location.origin)) {
    return;
  }

  const url = new URL(event.request.url);

  // Bypass third-party analytics, beacon, or challenge endpoints
  if (url.hostname.includes('cloudflareinsights.com') || url.hostname.includes('google-analytics.com') || url.pathname.includes('/cdn-cgi/')) {
    return;
  }

  // Never cache the service worker script itself so browser can check for updates in background
  if (url.pathname.endsWith('/sw.js')) {
    event.respondWith(fetch(event.request));
    return;
  }

  const isNavigation = event.request.mode === 'navigate' || event.request.destination === 'document' || url.pathname.endsWith('.html') || url.pathname === '/';

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      // 1. Check cache first
      const cachedResponse = await cache.match(event.request);

      // 2. Background network fetch & cache update (double-check network for updated files)
      const networkFetch = fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          cache.put(event.request, networkResponse.clone());
        }
        return networkResponse;
      }).catch(() => null);

      if (cachedResponse) {
        // Cache hit: serve cached response immediately, and revalidate in background
        event.waitUntil(networkFetch);
        return cachedResponse;
      }

      // Cache miss: wait for network response
      const networkResponse = await networkFetch;
      if (networkResponse) {
        return networkResponse;
      }

      // Offline fallback on cache miss
      if (isNavigation) {
        const fallback = await cache.match('/index.html') || await cache.match('/');
        if (fallback) {
          return fallback;
        }
      }

      return new Response('Offline: Network unavailable', {
        status: 503,
        statusText: 'Service Unavailable',
        headers: new Headers({ 'Content-Type': 'text/plain; charset=utf-8' })
      });
    })
  );
});
