/**
 * TCS volatility tracker on Cloudflare.
 *
 *   cron (weekdays 17:40 UTC = 23:10 IST) ─► Runner (Durable Object) ─► container: python -m finbert_vol.live serve
 *        R2 state/state.tar.gz ──POST /run──► container ──JSON──► R2 public/*.{csv,json} + new state
 *   GET /             static dashboard (public/index.html)
 *   GET /api/<file>   tracker.csv | latest.json | summary.json | headlines.json   (from R2)
 *   GET /api/status   last run log
 *   POST /api/run     manual run (Authorization: Bearer <RUN_TOKEN>)
 */
import { DurableObject } from "cloudflare:workers";

interface Env {
  RUNNER: DurableObjectNamespace<Runner>;
  BUCKET: R2Bucket;
  ASSETS: Fetcher;
  RUN_TOKEN?: string;
}

interface RunResult { ok: boolean; error: string | null; log: string[]; latest: unknown; public: Record<string, string>; state_b64: string | null }

const PUBLIC = new Set(["tracker.csv", "latest.json", "summary.json", "headlines.json"]);
const TYPES: Record<string, string> = { csv: "text/csv; charset=utf-8", json: "application/json; charset=utf-8" };

export class Runner extends DurableObject<Env> {
  private current: Promise<{ ok: boolean; error: string | null; started: string; seconds: number }> | undefined;

  /** One run at a time: a second trigger while running waits for the same run. */
  run(reason: string) {
    this.current ??= this.runOnce(reason).finally(() => { this.current = undefined; });
    return this.current;
  }

  private async runOnce(reason: string) {
    const started = new Date(); const t0 = Date.now();
    const container = this.ctx.container!;
    if (!container.running) container.start({ enableInternet: true });
    const port = container.getTcpPort(8080);

    // wait for the HTTP server (model loads lazily on the first /run, so this is quick)
    let lastError: unknown;
    for (let i = 0; i < 300; i++) {
      try { const r = await port.fetch("http://container/"); if (r.ok) { lastError = undefined; break; } }
      catch (e) { lastError = e; }
      await scheduler.wait(200);
    }
    if (lastError) throw lastError;

    const state = await this.env.BUCKET.get("state/state.tar.gz");
    const body = state ? await state.arrayBuffer() : new ArrayBuffer(0);
    let res: RunResult;
    try {
      const resp = await port.fetch("http://container/run", { method: "POST", body, headers: { "content-type": "application/gzip" } });
      res = await resp.json<RunResult>();
    } catch (e) {
      res = { ok: false, error: `container call failed: ${e}`, log: [], latest: null, public: {}, state_b64: null };
    }

    if (res.ok && res.state_b64) {
      // keep yesterday's state as a fallback, then publish
      if (state) await this.env.BUCKET.put("state/state.prev.tar.gz", body);
      await this.env.BUCKET.put("state/state.tar.gz", Uint8Array.from(atob(res.state_b64), (c) => c.charCodeAt(0)));
      for (const [name, text] of Object.entries(res.public)) {
        if (PUBLIC.has(name)) await this.env.BUCKET.put(`public/${name}`, text, { httpMetadata: { contentType: TYPES[name.split(".").pop()!] } });
      }
    }
    const summary = { ok: res.ok, error: res.error, started: started.toISOString(), seconds: Math.round((Date.now() - t0) / 1000) };
    await this.env.BUCKET.put("logs/last.json", JSON.stringify({ reason, ...summary, latest: res.latest, log: res.log }, null, 1),
      { httpMetadata: { contentType: TYPES.json } });
    await this.env.BUCKET.put(`logs/${started.toISOString().slice(0, 10)}.json`, JSON.stringify({ reason, ...summary, log: res.log }));
    container.signal?.(15);   // SIGTERM: done for today (the platform also stops it when idle)
    return summary;
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/api/run" && req.method === "POST") {
      const auth = req.headers.get("authorization") ?? "";
      if (!env.RUN_TOKEN || auth !== `Bearer ${env.RUN_TOKEN}`) return new Response("unauthorized", { status: 401 });
      const r = await env.RUNNER.getByName("daily").run("manual");
      return Response.json(r, { status: r.ok ? 200 : 500 });
    }
    if (url.pathname === "/api/status") {
      const o = await env.BUCKET.get("logs/last.json");
      return o ? new Response(o.body, { headers: { "content-type": TYPES.json, "cache-control": "no-store" } }) : Response.json({ ok: null, note: "no run yet" });
    }
    if (url.pathname.startsWith("/api/")) {
      const name = url.pathname.slice(5);
      if (!PUBLIC.has(name)) return new Response("not found", { status: 404 });
      const o = await env.BUCKET.get(`public/${name}`);
      if (!o) return new Response("not published yet", { status: 404 });
      return new Response(o.body, { headers: { "content-type": TYPES[name.split(".").pop()!], "cache-control": "public, max-age=300" } });
    }
    return env.ASSETS.fetch(req);
  },

  async scheduled(_c: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(env.RUNNER.getByName("daily").run("cron"));
  },
} satisfies ExportedHandler<Env>;
