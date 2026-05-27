import { useMemo, useState } from "react";
import Navbar from "../components/Navbar";

const MODELS = [
  { value: "xgboost_residual", label: "XGBoost (Residual Learning) — Best" },
  { value: "lstm", label: "LSTM" },
  { value: "tcn", label: "TCN" },
  { value: "transformer", label: "Transformer (Temporal Encoder)" },
];

type PlotItem = { name: string; b64: string };

export default function SingleSite() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState(MODELS[1].value);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
  const [plots, setPlots] = useState<PlotItem[]>([]);

  const canRun = useMemo(
    () => !!file && !!model && !loading,
    [file, model, loading],
  );
  const apiBase = useMemo(() => {
    const base = (import.meta.env.VITE_API_BASE as string | undefined) || "";
    return base.endsWith("/") ? base.slice(0, -1) : base;
  }, []);

  async function run() {
    if (!file) return;

    setLoading(true);
    setErr(null);
    setMetrics(null);
    setPlots([]);

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("model_name", model);

      const res = await fetch(`${apiBase}/api/single-site/run`, {
        method: "POST",
        body: fd,
      });

      // if backend crashes / returns HTML, this avoids "Unexpected token <"
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await res.text();
        throw new Error(
          `Backend did not return JSON (status ${res.status}). Response: ${text.slice(0, 200)}`,
        );
      }

      const data = await res.json();

      if (!res.ok || !data.ok) {
        // show useful backend error if available
        const msg =
          data?.error || data?.detail || `Request failed (HTTP ${res.status})`;
        const stderr = data?.stderr || data?.logs?.stderr_tail;
        const stdout = data?.stdout || data?.logs?.stdout_tail;
        const extra = [stderr, stdout].filter(Boolean).join("\n\n");
        throw new Error(extra ? `${msg}\n\n${extra}` : msg);
      }

      setMetrics(data.metrics || {});
      setPlots(Array.isArray(data.plots) ? data.plots : []);

      // auto scroll to results
      setTimeout(() => {
        document
          .getElementById("results")
          ?.scrollIntoView({ behavior: "smooth" });
      }, 150);
    } catch (e: any) {
      setErr(e?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="px-6 md:px-12 py-14">
      <Navbar />
      <div className="mx-auto max-w-6xl">
        <div className="text-xs tracking-[0.3em] text-cyan-300/80">
          <br></br>
          <br></br>
          <br></br>
          <br></br>
          SINGLE-SITE
        </div>

        <div className="mt-4 grid gap-10 lg:grid-cols-2 lg:items-start">
          {/* Left: Title + explanation */}
          <div>
            <h1 className="text-3xl md:text-5xl font-semibold leading-tight">
              Run Stratowatch On Single Dataset
            </h1>
            <p className="mt-4 text-white/60 max-w-xl leading-relaxed">
              Upload a CSV/XLSX, choose a model, then run evaluation to generate
              verified metrics and plots (loss curves, heatmaps, confusion
              matrices, and more).
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-5 glow-hover">
                <div className="text-white/80 font-medium">Inputs</div>
                <div className="mt-2 text-white/55 text-sm">
                  CSV/XLSX dataset with required columns (same schema as
                  training).
                </div>
              </div>
              <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-5 glow-hover">
                <div className="text-white/80 font-medium">Outputs</div>
                <div className="mt-2 text-white/55 text-sm">
                  MAE / RMSE / R² + model artifacts & plots returned below.
                </div>
              </div>
            </div>
          </div>

          {/* Right: Premium run card */}
          <div className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-8 glow-hover relative overflow-hidden">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.18),transparent_60%)]" />
            <div className="relative grid gap-6">
              {/* Upload */}
              <div>
                <div className="text-white/80 font-medium">Dataset</div>
                <label className="mt-3 block rounded-2xl bg-white/5 ring-1 ring-white/10 p-6 glow-hover cursor-pointer hover:bg-white/10 transition">
                  <input
                    type="file"
                    accept=".csv,.xlsx"
                    className="hidden"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                  <div className="text-white/60 text-sm">
                    {file ? (
                      <>
                        <div className="text-white font-medium">
                          {file.name}
                        </div>
                        <div className="mt-1">Click to replace</div>
                      </>
                    ) : (
                      "Click to select CSV/XLSX (drag-drop can be added next)"
                    )}
                  </div>
                </label>
              </div>

              {/* Model select */}
              <div>
                <div className="text-white/80 font-medium">Model</div>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="mt-3 w-full rounded-2xl bg-black/40 ring-1 ring-white/10 px-4 py-3 text-white/80"
                >
                  {MODELS.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>

                <button
                  disabled={!canRun}
                  onClick={run}
                  className="mt-4 w-full rounded-2xl bg-cyan-400 px-5 py-3 text-black font-semibold disabled:opacity-40 hover:bg-cyan-300 transition"
                >
                  {loading ? "Running..." : "Run Model"}
                </button>

                {err && <div className="mt-3 text-sm text-red-300">{err}</div>}
              </div>

              <div className="rounded-2xl bg-black/30 ring-1 ring-white/10 p-5 glow-hover text-sm text-white/55">
                Tip: results will appear below after the run completes.
              </div>
            </div>
          </div>
        </div>

        {/* RESULTS (below) */}
        <div id="results" className="mt-14">
          <div className="text-xs tracking-[0.3em] text-cyan-300/80">
            RESULTS
          </div>
          <h2 className="mt-3 text-2xl md:text-3xl font-semibold">
            Metrics & plots
          </h2>
          <p className="mt-3 text-white/60 max-w-3xl">
            After execution, metrics are shown in cards and plots are rendered
            as images below.
          </p>

          {!metrics && plots.length === 0 ? (
            <div className="mt-8 rounded-3xl bg-white/5 ring-1 ring-white/10 p-10 glow-hover text-white/50">
              No results yet. Upload a file and click{" "}
              <span className="text-white/80">Run Model</span>.
            </div>
          ) : (
            <>
              {/* Metric cards */}
              <div className="mt-8 grid gap-6 md:grid-cols-3">
                {metrics && "MAE" in metrics && (
                  <div className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-7 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(56,189,248,0.18)] transition glow-hover">
                    <div className="text-white/60 text-sm">MAE</div>
                    <div className="mt-3 text-4xl font-semibold">
                      {metrics.MAE}
                    </div>
                  </div>
                )}
                {metrics && "RMSE" in metrics && (
                  <div className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-7 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(56,189,248,0.18)] transition glow-hover">
                    <div className="text-white/60 text-sm">RMSE</div>
                    <div className="mt-3 text-4xl font-semibold">
                      {metrics.RMSE}
                    </div>
                  </div>
                )}
                {metrics && "R2" in metrics && (
                  <div className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-7 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(56,189,248,0.18)] transition glow-hover">
                    <div className="text-white/60 text-sm">R²</div>
                    <div className="mt-3 text-4xl font-semibold">
                      {metrics.R2}
                    </div>
                  </div>
                )}
              </div>

              {/* Plots */}
              {plots.length > 0 && (
                <div className="mt-10 grid gap-6 md:grid-cols-2">
                  {plots.map((p) => (
                    <div
                      key={p.name}
                      className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-5 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(56,189,248,0.18)] transition glow-hover"
                    >
                      <div className="text-white/70 text-sm mb-3">{p.name}</div>
                      <img
                        src={`data:image/png;base64,${p.b64}`}
                        className="w-full rounded-2xl ring-1 ring-white/10"
                        alt={p.name}
                      />
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
