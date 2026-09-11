
import { useMemo, useState } from "react";

import Navbar from "../components/Navbar";

const MODELS = [
  { value: "st_transformer", label: "ST Transformer (Best MAE)" },
  { value: "graph_st_static", label: "Static Graph-ST" },
  { value: "graph_st_dynamic_wind", label: "Dynamic Wind Graph-ST" },
];

type PlotItem = {
  name: string;
  b64: string;
};

type MultiSiteResponse = {
  ok?: boolean;
  success?: boolean;
  mode?: string;
  inference_mode?: string;
  model?: string;
  site_count?: number;
  sites?: string[];
  metrics?: Record<string, any>;
  plots?: PlotItem[];
  warning?: string;
  warnings?: string[];
  error?: string;
  detail?: string;
  stdout?: string;
  stderr?: string;
  logs?: {
    stdout_tail?: string;
    stderr_tail?: string;
  };
  uploaded_file_count?: number;
  uploaded_files_used_for_research_evaluation?: boolean;
};

export default function MultiSite() {
  const [files, setFiles] = useState<File[]>([]);
  const [siteCount, setSiteCount] = useState<number>(7);
  const [model, setModel] = useState(MODELS[0].value);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
  const [plots, setPlots] = useState<PlotItem[]>([]);
  const [sites, setSites] = useState<string[]>([]);
  const [inferenceMode, setInferenceMode] = useState<string | null>(null);
  const [
    uploadedFilesUsedForResearchEvaluation,
    setUploadedFilesUsedForResearchEvaluation,
  ] = useState<boolean | null>(null);

  const fmt = (v: any, digits = 4) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return n.toFixed(digits);
  };

  const canRun = useMemo(() => {
    if (loading) return false;
    if (!model) return false;
    if (siteCount < 2) return false;
    if (files.length !== siteCount) return false;
    return true;
  }, [files, siteCount, model, loading]);

  const apiBase = useMemo(() => {
    const base =
      (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
      "http://127.0.0.1:8000";

    return base.replace(/\/+$/, "");
  }, []);

  function resetResults() {
    setErr(null);
    setWarning(null);
    setMetrics(null);
    setPlots([]);
    setSites([]);
    setInferenceMode(null);
    setUploadedFilesUsedForResearchEvaluation(null);
  }

  function isCsv(file: File): boolean {
    return file.name.toLowerCase().endsWith(".csv");
  }

  // Add selected files until siteCount is reached.
  function addFiles(list: FileList | null) {
    if (!list) return;

    const incoming = Array.from(list).filter(isCsv);

    setFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}_${f.size}`));
      const merged = [...prev];

      for (const f of incoming) {
        const key = `${f.name}_${f.size}`;

        if (!seen.has(key)) {
          merged.push(f);
          seen.add(key);
        }

        if (merged.length >= siteCount) break;
      }

      return merged;
    });

    resetResults();
  }

  // Replace all files with a new CSV selection.
  function replaceAll(list: FileList | null) {
    if (!list) return;

    const incoming = Array.from(list)
      .filter(isCsv)
      .slice(0, siteCount);

    setFiles(incoming);
    resetResults();
  }

  function removeFile(idx: number) {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
    resetResults();
  }

  function clearAll() {
    setFiles([]);
    resetResults();
  }

  async function run() {
    if (!canRun) return;

    setLoading(true);
    setErr(null);
    setWarning(null);
    setMetrics(null);
    setPlots([]);
    setSites([]);
    setInferenceMode(null);

    try {
      const fd = new FormData();

      files.forEach((file) => {
        fd.append("files", file);
      });

      fd.append("model_name", model);
      fd.append("site_count", String(siteCount));

      const res = await fetch(`${apiBase}/api/multi-site/run`, {
        method: "POST",
        body: fd,
      });

      const contentType = res.headers.get("content-type") || "";

      if (!contentType.includes("application/json")) {
        const text = await res.text();

        throw new Error(
          `Backend did not return JSON (HTTP ${res.status}). ${text.slice(
            0,
            200,
          )}`,
        );
      }

      const data = (await res.json()) as MultiSiteResponse;

      if (!res.ok || data.ok === false || data.success === false) {
        const msg =
          data.error || data.detail || `Request failed (HTTP ${res.status})`;

        const stderr = data.stderr || data.logs?.stderr_tail;
        const stdout = data.stdout || data.logs?.stdout_tail;
        const extra = [stderr, stdout].filter(Boolean).join("\n\n");

        throw new Error(extra ? `${msg}\n\n${extra}` : msg);
      }

      const combinedWarnings = [
        ...(data.warning ? [data.warning] : []),
        ...(data.warnings || []),
      ];

      setWarning(
        combinedWarnings.length > 0
          ? combinedWarnings.join("\n\n")
          : null,
      );

      setMetrics(data.metrics || {});
      setPlots(data.plots || []);
      setSites(data.sites || []);
      setInferenceMode(data.inference_mode || data.mode || null);

      setUploadedFilesUsedForResearchEvaluation(
        data.uploaded_files_used_for_research_evaluation ?? null,
      );

      setTimeout(() => {
        document
          .getElementById("results")
          ?.scrollIntoView({ behavior: "smooth" });
      }, 150);
    } catch (e) {
      let message: string;

      if (
        e instanceof TypeError &&
        (e.message === "Failed to fetch" ||
          e.message.includes("NetworkError"))
      ) {
        message =
          "Cannot reach the StratoWatch backend.\n\n" +
          "The backend API (FastAPI / Python) must be running and accessible.\n\n" +
          "• Local development: run `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` in the backend/ directory.\n" +
          "• Production: the backend must be deployed to a cloud service (e.g. Railway, Render) and VITE_API_BASE_URL must point to it.";
      } else {
        message = e instanceof Error ? e.message : "Something went wrong";
      }

      setErr(message);
    } finally {
      setLoading(false);
    }
  }

  const remaining = Math.max(0, siteCount - files.length);

  return (
    <div className="px-6 md:px-12 py-14">
      <Navbar />

      <div className="mx-auto max-w-6xl">
        <div className="text-xs tracking-[0.3em] text-cyan-300/80">
          <br />
          <br />
          <br />
          <br />
          MULTI-SITE
        </div>

        <div className="mt-4 grid gap-10 lg:grid-cols-2 lg:items-start">
          {/* LEFT */}
          <div>
            <h1 className="text-3xl md:text-5xl font-semibold leading-tight">
              Run Stratowatch On Multiple Dataset
            </h1>

            <p className="mt-4 text-white/60 max-w-xl leading-relaxed">
              Upload one dataset per site, choose a spatio-temporal model
              variant, and generate full evaluation analysis including
              horizon-wise errors, residual distributions, confusion bins,
              embeddings, and prediction plots.
            </p>

            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-5 glow-hover">
                <div className="text-white/80 font-medium">Inputs</div>

                <div className="mt-2 text-white/55 text-sm">
                  Multiple CSV files (one per station) + site count.
                </div>
              </div>

              <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-5 glow-hover">
                <div className="text-white/80 font-medium">Outputs</div>

                <div className="mt-2 text-white/55 text-sm">
                  MAE / RMSE (+ optional per-site & horizon metrics) and figures
                  below.
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL */}
          <div className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-8 glow-hover relative overflow-hidden">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.18),transparent_60%)]" />

            <div className="relative grid gap-6">
              {/* SITE COUNT */}
              <div>
                <div className="text-white/80 font-medium">
                  How many sites?
                </div>

                <input
                  type="number"
                  min={2}
                  value={siteCount}
                  onChange={(e) => {
                    const n = Number(e.target.value);

                    setSiteCount(n);
                    setFiles((prev) =>
                      prev.slice(0, Math.max(0, n)),
                    );
                    resetResults();
                  }}
                  className="mt-3 w-full rounded-2xl bg-black/40 ring-1 ring-white/10 px-4 py-3 text-white/80"
                />

                {/* 7-sites recommendation */}
                {siteCount === 7 ? (
                  <div className="mt-3 text-sm text-emerald-400">
                    ✅ Recommended — frozen research checkpoints are available
                    for 7 sites.
                  </div>
                ) : (
                  <div className="mt-3 text-sm text-cyan-300/80">
                    ℹ {siteCount}-site requests use request-local forecast
                    mode. The frozen research checkpoints are available for 7
                    sites.
                  </div>
                )}

                <div className="mt-2 text-xs text-white/45">
                  Required files:{" "}
                  <span className="text-white/70">{siteCount}</span> |
                  Selected:{" "}
                  <span className="text-white/70">{files.length}</span>
                </div>
              </div>

              {/* FILE UPLOAD */}
              <div>
                <div className="text-white/80 font-medium">
                  Datasets (multiple)
                </div>

                <div className="mt-3 rounded-2xl bg-white/5 ring-1 ring-white/10 p-6 glow-hover">
                  {/* Main upload */}
                  <label className="block cursor-pointer">
                    <input
                      type="file"
                      accept=".csv"
                      multiple
                      className="hidden"
                      onChange={(e) => addFiles(e.target.files)}
                    />

                    <div className="text-white/60 text-sm">
                      {files.length === 0 ? (
                        "Click to add datasets (CSV)"
                      ) : remaining > 0 ? (
                        <>
                          <div className="text-white/80 font-medium">
                            Click to add more files ({remaining} remaining)
                          </div>

                          <div className="mt-1 text-white/55">
                            Keep adding until you reach {siteCount}.
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="text-white/80 font-medium">
                            All sites uploaded ✅
                          </div>

                          <div className="mt-1 text-white/55">
                            You can still replace all if you want.
                          </div>
                        </>
                      )}
                    </div>
                  </label>

                  {/* File list */}
                  {files.length > 0 && (
                    <div className="mt-4 grid gap-2">
                      {files.map((f, i) => (
                        <div
                          key={`${f.name}_${f.size}_${i}`}
                          className="flex items-center justify-between gap-3 rounded-xl bg-black/30 ring-1 ring-white/10 px-4 py-3"
                        >
                          <div className="min-w-0">
                            <div className="text-white/80 text-sm truncate">
                              {i + 1}. {f.name}
                            </div>

                            <div className="text-white/40 text-xs">
                              {(f.size / 1024).toFixed(1)} KB
                            </div>
                          </div>

                          <button
                            type="button"
                            onClick={() => removeFile(i)}
                            className="shrink-0 rounded-lg bg-white/10 px-3 py-1.5 text-xs text-white/70 ring-1 ring-white/10 hover:bg-white/15 hover:text-white transition"
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Actions row */}
                  <div className="mt-4 flex flex-wrap gap-2">
                    {/* Add more */}
                    <label className="cursor-pointer rounded-xl bg-cyan-400/20 px-4 py-2 text-xs text-cyan-300 ring-1 ring-cyan-400/40 hover:bg-cyan-400/30 transition">
                      <input
                        type="file"
                        accept=".csv"
                        multiple
                        className="hidden"
                        onChange={(e) => addFiles(e.target.files)}
                      />
                      Add more
                    </label>

                    {/* Replace all */}
                    <label className="cursor-pointer rounded-xl bg-white/10 px-4 py-2 text-xs text-white/70 ring-1 ring-white/10 hover:bg-white/15 hover:text-white transition">
                      <input
                        type="file"
                        accept=".csv"
                        multiple
                        className="hidden"
                        onChange={(e) => replaceAll(e.target.files)}
                      />
                      Replace all
                    </label>

                    {/* Clear */}
                    <button
                      type="button"
                      onClick={clearAll}
                      className="rounded-xl bg-white/10 px-4 py-2 text-xs text-white/70 ring-1 ring-white/10 hover:bg-white/15 hover:text-white transition"
                    >
                      Clear
                    </button>
                  </div>

                  {/* Count warning */}
                  {files.length > 0 && files.length !== siteCount && (
                    <div className="mt-3 text-xs text-amber-200">
                      You selected {files.length} file(s) but site count is{" "}
                      {siteCount}. Add {remaining} more.
                    </div>
                  )}
                </div>
              </div>

              {/* MODEL SELECT */}
              <div>
                <div className="text-white/80 font-medium">
                  Model variant
                </div>

                <select
                  value={model}
                  onChange={(e) => {
                    setModel(e.target.value);
                    resetResults();
                  }}
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

                {err && (
                  <div className="mt-3 text-sm text-red-300 whitespace-pre-line">
                    {err}
                  </div>
                )}
              </div>

              <div className="rounded-2xl bg-black/30 ring-1 ring-white/10 p-5 glow-hover text-sm text-white/55">
                Results will render below after the run completes.
              </div>
            </div>
          </div>
        </div>

        {/* WARNING BANNER */}
        {warning && (
          <div className="mt-6 rounded-2xl bg-cyan-300/5 ring-1 ring-cyan-300/20 p-5 text-cyan-100 text-sm whitespace-pre-line">
            {warning}
          </div>
        )}

        {/* RESULTS */}
        <div id="results" className="mt-14">
          <div className="text-xs tracking-[0.3em] text-cyan-300/80">
            RESULTS
          </div>

          <h2 className="mt-3 text-2xl md:text-3xl font-semibold">
            Metrics & Analysis
          </h2>

          {loading ? (
            <div className="mt-8 rounded-3xl bg-white/5 ring-1 ring-white/10 p-10 glow-hover text-white/50">
              Running the current request. Previous metrics and plots have
              been cleared.
            </div>
          ) : !metrics && plots.length === 0 ? (
            <div className="mt-8 rounded-3xl bg-white/5 ring-1 ring-white/10 p-10 glow-hover text-white/50">
              No results yet. Upload files and click{" "}
              <span className="text-white/80">Run Model</span>.
            </div>
          ) : (
            <>
              {(inferenceMode || sites.length > 0) && (
                <div className="mt-6 text-sm text-white/55">
                  {inferenceMode && (
                    <div>
                      Inference mode:{" "}
                      <span className="text-white/80">
                        {inferenceMode}
                      </span>
                    </div>
                  )}

                  {sites.length > 0 && (
                    <div className="mt-1">
                      Sites in this request:{" "}
                      <span className="text-white/80">
                        {sites.join(", ")}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Metric cards */}
              {metrics && (
                <div className="mt-8 grid gap-6 md:grid-cols-3">
                  {"MAE" in metrics && (
                    <MetricCard
                      label="MAE"
                      value={fmt(metrics.MAE)}
                    />
                  )}

                  {"RMSE" in metrics && (
                    <MetricCard
                      label="RMSE"
                      value={fmt(metrics.RMSE)}
                    />
                  )}

                  {"R2" in metrics && (
                    <MetricCard
                      label="R²"
                      value={fmt(metrics.R2)}
                    />
                  )}

                  {"mae" in metrics && !("MAE" in metrics) && (
                    <MetricCard
                      label="MAE"
                      value={fmt(metrics.mae)}
                    />
                  )}

                  {"rmse" in metrics && !("RMSE" in metrics) && (
                    <MetricCard
                      label="RMSE"
                      value={fmt(metrics.rmse)}
                    />
                  )}

                  {"r2" in metrics && !("R2" in metrics) && (
                    <MetricCard
                      label="R²"
                      value={fmt(metrics.r2)}
                    />
                  )}
                </div>
              )}

              {/* Plots */}
              {plots.length > 0 && (
                <div className="mt-10 grid gap-6 md:grid-cols-2">
                  {plots.map((p) => (
                    <div
                      key={p.name}
                      className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-5 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(56,189,248,0.18)] transition glow-hover"
                    >
                      <div className="text-white/70 text-sm mb-3 wrap-break-word">
                        {p.name}
                      </div>

                      <img
                        src={`data:image/png;base64,${p.b64}`}
                        className="w-full rounded-2xl ring-1 ring-white/10"
                        alt={p.name}
                        loading="lazy"
                      />
                    </div>
                  ))}
                </div>
              )}

              {/* Research evaluation source */}
              {metrics &&
                uploadedFilesUsedForResearchEvaluation === false && (
                  <div className="mt-8 rounded-2xl bg-cyan-300/5 ring-1 ring-cyan-300/15 p-5">
                    <div className="text-sm font-medium text-cyan-100">
                      Research evaluation source
                    </div>

                    <p className="mt-2 text-sm leading-6 text-white/55">
                      The results shown above come from the frozen StratoWatch
                      research evaluation workflow. The uploaded CSV files are
                      accepted by the interface but are not substituted for
                      the frozen research dataset.
                    </p>
                  </div>
                )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-3xl bg-white/5 ring-1 ring-white/10 p-7 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(56,189,248,0.18)] transition glow-hover">
      <div className="text-white/60 text-sm">{label}</div>

      <div className="mt-3 text-3xl md:text-4xl font-semibold tabular-nums wrap-break-word">
        {value}
      </div>
    </div>
  );
}

