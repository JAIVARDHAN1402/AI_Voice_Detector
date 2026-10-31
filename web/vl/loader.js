// Robust downloader for the big model files.
//  * parallel ranged requests (4 MB pieces, 6 at a time): one slow connection no longer stalls the whole file,
//  * a piece that makes no progress for `stallMs` is aborted and retried with back-off (this is the "stuck at 4/46 MB" case),
//  * falls back to one streaming request (also with stall detection + retries) if the server does not support Range,
//  * the finished file is kept in Cache Storage (ranged 206 responses are not cached by the HTTP cache), so the next visit is instant.
const CACHE = 'vl-models-v1';
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function cacheGet(key) {
  try { const c = await caches.open(CACHE); const hit = await c.match(key); return hit ? new Uint8Array(await hit.arrayBuffer()) : null; } catch (e) { return null; }
}
async function cachePut(key, bytes) {
  try { const c = await caches.open(CACHE); await c.put(key, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) { /* quota or private mode */ }
}

/** Read a response body, aborting if no bytes arrive for stallMs. `onBytes(n)` is called per piece of data. */
async function readWithStall(res, ctrl, stallMs, onBytes) {
  const reader = res.body.getReader();
  const parts = []; let got = 0;
  let timer = setTimeout(() => ctrl.abort(), stallMs);
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length; onBytes && onBytes(value.length);
      clearTimeout(timer); timer = setTimeout(() => ctrl.abort(), stallMs);
    }
  } finally { clearTimeout(timer); }
  const out = new Uint8Array(got); let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

async function getRange(url, start, end, o, onBytes) {
  let lastErr;
  for (let attempt = 0; attempt < o.retries; attempt++) {
    const ctrl = new AbortController();
    let counted = 0;
    try {
      const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` }, signal: ctrl.signal, cache: 'no-store' });
      if (res.status !== 206) throw new Error('range not honoured (' + res.status + ')');
      const bytes = await readWithStall(res, ctrl, o.stallMs, n => { counted += n; onBytes(n); });
      if (bytes.length !== end - start + 1) throw new Error('short piece');
      return bytes;
    } catch (e) {
      lastErr = e; onBytes(-counted);                       // un-count what this failed attempt had added
      await sleep(Math.min(8000, 400 * 2 ** attempt));
    }
  }
  throw lastErr || new Error('download failed');
}

async function streamWhole(url, o, onProgress, expected) {
  let lastErr;
  for (let attempt = 0; attempt < o.retries; attempt++) {
    const ctrl = new AbortController();
    try {
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const total = Number(res.headers.get('content-length')) || expected || 0;
      let got = 0;
      return await readWithStall(res, ctrl, o.stallMs, n => { got += n; onProgress && onProgress(got, total); });
    } catch (e) { lastErr = e; await sleep(Math.min(8000, 400 * 2 ** attempt)); }
  }
  throw lastErr || new Error('download failed');
}

/** -> Uint8Array. `key` is the cache key (use a content-hashed URL so a stale file can never be served). */
export async function fetchRobust(url, { onProgress, expected = 0, key = url, chunk = 4 * 1024 * 1024, parallel = 6, stallMs = 15000, retries = 6 } = {}) {
  const hit = await cacheGet(key);
  if (hit && (!expected || hit.length === expected)) { onProgress && onProgress(hit.length, hit.length); return hit; }
  const o = { chunk, parallel, stallMs, retries };
  let total = 0;
  try {                                                      // probe: does the server do Range, and how big is the file?
    const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 20000);
    const r = await fetch(url, { headers: { Range: 'bytes=0-0' }, signal: ctrl.signal, cache: 'no-store' });
    clearTimeout(t);
    if (r.status === 206) total = Number((r.headers.get('content-range') || '').split('/')[1]) || 0;
    try { await r.arrayBuffer(); } catch (e) { /* ignore */ }
  } catch (e) { /* fall through to the streaming path */ }
  let out;
  if (total > 0) {
    out = new Uint8Array(total);
    let done = 0; const n = Math.ceil(total / chunk); let next = 0;
    const bump = d => { done += d; onProgress && onProgress(Math.max(0, done), total); };
    const worker = async () => {
      for (;;) {
        const i = next++; if (i >= n) return;
        const s = i * chunk, e = Math.min(total, s + chunk) - 1;
        out.set(await getRange(url, s, e, o, bump), s);
      }
    };
    await Promise.all(Array.from({ length: Math.min(parallel, n) }, worker));
  } else {
    out = await streamWhole(url, o, onProgress, expected);
  }
  if (expected && out.length !== expected) throw new Error(`downloaded ${out.length} bytes, expected ${expected}`);
  await cachePut(key, out);
  return out;
}
