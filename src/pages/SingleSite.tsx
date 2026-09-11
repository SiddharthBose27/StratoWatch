import { useMemo, useState } from "react";
import Navbar from "../components/Navbar";
import { ui } from "../styles/ui";

type MetricValue = number | null | undefined;

type ModelMetrics = {
  overall?: {
    mae?: MetricValue;
    rmse?: MetricValue;
    r2?: MetricValue;
  };
  per_target?: Record<string, unknown>;
  horizon_wise?: unknown;
  raw?: Record<string, unknown>;
};

type PlotItem = {
  name: string;
  b64: string;
};

type RunResponse = {
  ok?: boolean;
  success?: boolean;
  model?: string;
  metrics?: ModelMetrics;
  plots?: PlotItem[];
  warnings?: string[];
  warning?: string;
  stdout?: string;
  stderr?: string;
  error?: string;
  detail?: string;
  evaluation_source?: string;
  uploaded_file_used_for_research_evaluation?: boolean;
  evaluation_only?: boolean;
  retraining?: boolean;
};

type ModelOption = {
  value: string;
  label: string;
  description: string;
};

const MODELS: ModelOption[] = [
  {
    value: "xgboost",
    label: "XGBoost",
    description: "Best single-site performance",
  },
  {
    value: "random_forest",
    label: "Random Forest",
    description: "Tree-based ensemble baseline",
  },
  {
    value: "tcn",
    label: "TCN",
    description: "Temporal convolutional model",
  },
  {
    value: "transformer",
    label: "Transformer",
    description: "Temporal attention model",
  },
  {
    value: "lstm",
    label: "LSTM",
    description: "Recurrent temporal baseline",
  },
];

const PLOT_LABELS: Record<string, string> = {
  o3_actual_vs_predicted: "O3 — Actual vs Predicted",
  no2_actual_vs_predicted: "NO2 — Actual vs Predicted",

  o3_scatter: "O3 — Prediction Scatter",
  no2_scatter: "NO2 — Prediction Scatter",

  o3_residual_distribution: "O3 — Residual Distribution",
  no2_residual_distribution: "NO2 — Residual Distribution",

  horizon_performance: "Forecast Horizon — MAE / RMSE",
  o3_error_by_horizon: "O3 — Error by Forecast Horizon",
  no2_error_by_horizon: "NO2 — Error by Forecast Horizon",

  target_performance: "Target-wise Performance",
};

const PLOT_GROUPS = [
  {
    title: "Forecast behaviour",
    description:
      "How the model's forecasts compare with the observed O3 and NO2 values.",
    keys: ["o3_actual_vs_predicted", "no2_actual_vs_predicted"],
  },
  {
    title: "Prediction quality",
    description:
      "Scatter and residual views for checking accuracy, bias and error structure.",
    keys: [
      "o3_scatter",
      "no2_scatter",
      "o3_residual_distribution",
      "no2_residual_distribution",
    ],
  },
  {
    title: "Forecast horizon analysis",
    description:
      "How prediction error changes across the six forecast horizons.",
    keys: [
      "horizon_performance",
      "o3_error_by_horizon",
      "no2_error_by_horizon",
    ],
  },
  {
    title: "Target comparison",
    description: "Comparison of performance across the two forecast targets.",
    keys: ["target_performance"],
  },
];

function formatMetric(value: MetricValue): string {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(Number(value))
  ) {
    return "—";
  }

  return Number(value).toFixed(4);
}

function getApiBase(): string {
  const base =
    (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
    "http://127.0.0.1:8000";

  return base.replace(/\/+$/, "");
}

function normalisePlotUrl(url: string): string {
  if (!url) {
    return "";
  }

  if (
    url.startsWith("http://") ||
    url.startsWith("https://") ||
    url.startsWith("data:")
  ) {
    return url;
  }

  const backendBase = getApiBase();

  if (url.startsWith("/")) {
    return `${backendBase}${url}`;
  }

  return `${backendBase}/${url}`;
}

function getPlotTitle(key: string): string {
  return (
    PLOT_LABELS[key] ||
    key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase())
  );
}

function getPlotKey(plot: PlotItem): string {
  return plot.name.replace(/\.[^.]+$/, "");
}

function sortPlots(plots: PlotItem[]): PlotItem[] {
  const priority = [
    "o3_actual_vs_predicted",
    "no2_actual_vs_predicted",
    "o3_scatter",
    "no2_scatter",
    "o3_residual_distribution",
    "no2_residual_distribution",
    "horizon_performance",
    "o3_error_by_horizon",
    "no2_error_by_horizon",
    "target_performance",
  ];

  return [...plots].sort((a, b) => {
    const aKey = getPlotKey(a);
    const bKey = getPlotKey(b);

    const aIndex = priority.indexOf(aKey);
    const bIndex = priority.indexOf(bKey);

    if (aIndex === -1 && bIndex === -1) {
      return a.name.localeCompare(b.name);
    }

    if (aIndex === -1) {
      return 1;
    }

    if (bIndex === -1) {
      return -1;
    }

    return aIndex - bIndex;
  });
}

function MetricCard({
  label,
  value,
  emphasis = false,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div className={`${ui.metricBox} ${emphasis ? "ring-cyan-300/30" : ""}`}>
      <div className="text-xs tracking-[0.2em] text-white/45 uppercase">
        {label}
      </div>

      <div
        className={`mt-3 text-2xl md:text-3xl font-semibold ${
          emphasis ? "text-cyan-200" : "text-white"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function PlotCard({ plot }: { plot: PlotItem }) {
  const [failed, setFailed] = useState(false);

  const src = useMemo(() => {
    if (!plot.b64) {
      return "";
    }

    if (
      plot.b64.startsWith("data:") ||
      plot.b64.startsWith("http://") ||
      plot.b64.startsWith("https://") ||
      plot.b64.startsWith("/")
    ) {
      return normalisePlotUrl(plot.b64);
    }

    return `data:image/png;base64,${plot.b64}`;
  }, [plot.b64]);

  const plotKey = getPlotKey(plot);
  const title = getPlotTitle(plotKey);

  if (failed || !src) {
    return (
      <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-5">
        <div className="text-sm font-medium text-white/80">{title}</div>

        <div className="mt-4 rounded-xl bg-black/30 ring-1 ring-white/5 p-6 text-sm text-white/45">
          Plot could not be loaded.
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl bg-white/[0.035] ring-1 ring-white/10 overflow-hidden">
      <div className="px-5 py-4 border-b border-white/10">
        <div className="text-sm font-medium text-white/85">{title}</div>
      </div>

      <div className="bg-black/20 p-3">
        <img
          src={src}
          alt={title}
          className="w-full h-auto rounded-xl"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      </div>
    </div>
  );
}

export default function SingleSite() {
  const [selectedModel, setSelectedModel] = useState("xgboost");

  const [file, setFile] = useState<File | null>(null);

  const [result, setResult] = useState<RunResponse | null>(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const selectedModelInfo = MODELS.find(
    (model) => model.value === selectedModel,
  );

  const metrics = result?.metrics;

  const plots = useMemo(
    () => (result?.plots ? sortPlots(result.plots) : []),
    [result],
  );

  const plotEntries = plots.map((plot) => ({
    plot,
    key: getPlotKey(plot),
  }));

  async function runModel() {
    setError("");
    setResult(null);

    if (!file) {
      setError("Please select a CSV file before running the model.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();

      formData.append("model_name", selectedModel);
      formData.append("file", file);

      const response = await fetch(`${getApiBase()}/api/single-site/run`, {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type") || "";

      if (!contentType.includes("application/json")) {
        const text = await response.text();

        throw new Error(
          `Backend did not return JSON (HTTP ${response.status}). ${text.slice(
            0,
            200,
          )}`,
        );
      }

      const data = (await response.json()) as RunResponse;

      if (!response.ok || data.ok === false || data.success === false) {
        const message =
          data.error ||
          data.detail ||
          "Single-site forecasting request failed.";

        const extra = [data.stderr, data.stdout].filter(Boolean).join("\n\n");

        throw new Error(extra ? `${message}\n\n${extra}` : message);
      }

      setResult(data);
    } catch (err) {
      let message: string;

      if (err instanceof TypeError && (err.message === "Failed to fetch" || err.message.includes("NetworkError"))) {
        message =
          "Cannot reach the StratoWatch backend.\n\n" +
          "The backend API (FastAPI / Python) must be running and accessible.\n\n" +
          "• Local development: run `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` in the backend/ directory.\n" +
          "• Production: the backend must be deployed to a cloud service (e.g. Railway, Render) and VITE_API_BASE_URL must point to it.";
      } else {
        message = err instanceof Error
          ? err.message
          : "Something went wrong while running the model.";
      }

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] || null;

    if (selected && !selected.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setResult(null);
      setError("Please select a CSV file.");
      return;
    }

    setFile(selected);
    setResult(null);
    setError("");
  }

  return (
    <main className={ui.page}>
      <Navbar />

      <section className={`${ui.container} ${ui.section} pt-24`}>
        {/* -------------------------------------------------------------- */}
        {/* Header                                                         */}
        {/* -------------------------------------------------------------- */}

        <div className="max-w-3xl">
          <div className={ui.kLabel}>SINGLE-SITE FORECASTING</div>

          <h1 className="mt-4 text-4xl md:text-6xl font-semibold tracking-tight">
            Air-quality forecasting
          </h1>

          <p className="mt-6 text-base md:text-lg leading-8 text-white/60">
            Run a finalized StratoWatch single-site forecasting model for six
            forecast horizons across O3 and NO2.
          </p>
        </div>

        {/* -------------------------------------------------------------- */}
        {/* Controls                                                        */}
        {/* -------------------------------------------------------------- */}

        <div className="mt-12 grid lg:grid-cols-[1fr_360px] gap-6">
          <div className={`${ui.card} ${ui.glow}`}>
            <div className="relative">
              <div className={ui.kLabel}>MODEL</div>

              <div className="mt-5 grid sm:grid-cols-2 gap-3">
                {MODELS.map((model) => {
                  const active = selectedModel === model.value;

                  return (
                    <button
                      key={model.value}
                      type="button"
                      onClick={() => {
                        setSelectedModel(model.value);
                        setResult(null);
                        setError("");
                      }}
                      className={[
                        "text-left rounded-2xl p-5 ring-1 transition duration-200",
                        active
                          ? "bg-cyan-300/10 ring-cyan-300/40"
                          : "bg-white/5 ring-white/10 hover:bg-white/10 hover:ring-white/20",
                      ].join(" ")}
                    >
                      <div
                        className={`font-medium ${
                          active ? "text-cyan-200" : "text-white/85"
                        }`}
                      >
                        {model.label}
                      </div>

                      <div className="mt-2 text-sm text-white/45">
                        {model.description}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className={ui.card}>
            <div className={ui.kLabel}>INPUT</div>

            <div className="mt-5">
              <label
                htmlFor="single-site-file"
                className="block rounded-2xl border border-dashed border-white/15 bg-white/[0.03] p-6 cursor-pointer hover:bg-white/[0.06] transition"
              >
                <div className="text-sm font-medium text-white/80">
                  {file ? file.name : "Choose CSV"}
                </div>

                <div className="mt-2 text-xs leading-5 text-white/40">
                  Upload a CSV dataset for the single-site forecasting
                  interface. The backend determines whether the uploaded file is
                  used by the evaluation workflow.
                </div>

                <input
                  id="single-site-file"
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="sr-only"
                />
              </label>

              <button
                type="button"
                onClick={runModel}
                disabled={loading || !file}
                className="mt-4 w-full rounded-2xl bg-white text-black px-5 py-4 text-sm font-semibold transition hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading
                  ? "Running..."
                  : `Run ${selectedModelInfo?.label || "model"}`}
              </button>
            </div>
          </div>
        </div>

        {/* -------------------------------------------------------------- */}
        {/* Error                                                           */}
        {/* -------------------------------------------------------------- */}

        {error && (
          <div className="mt-8 rounded-2xl bg-red-400/10 ring-1 ring-red-300/20 p-5">
            <div className="text-sm font-medium text-red-200">
              Forecasting error
            </div>

            <div className="mt-2 text-sm leading-6 text-red-100/70 whitespace-pre-line">
              {error}
            </div>
          </div>
        )}

        {/* -------------------------------------------------------------- */}
        {/* Results                                                         */}
        {/* -------------------------------------------------------------- */}

        {result && (
          <section className="mt-16">
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
              <div>
                <div className={ui.kLabel}>FORECAST RESULTS</div>

                <h2 className={ui.h2}>{selectedModelInfo?.label}</h2>

                <p className="mt-3 text-sm text-white/45">
                  Six-horizon forecasting across O3 and NO2.
                </p>
              </div>

              {result.evaluation_source && (
                <div className="rounded-full bg-white/5 ring-1 ring-white/10 px-4 py-2 text-xs text-white/50">
                  Frozen test evaluation · {selectedModelInfo?.label}
                </div>
              )}
            </div>

            {/* ---------------------------------------------------------- */}
            {/* Metrics                                                     */}
            {/* ---------------------------------------------------------- */}

            <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
              <MetricCard
                label="MAE"
                value={formatMetric(metrics?.overall?.mae)}
                emphasis
              />

              <MetricCard
                label="RMSE"
                value={formatMetric(metrics?.overall?.rmse)}
              />

              <MetricCard
                label="R²"
                value={formatMetric(metrics?.overall?.r2)}
              />
            </div>

            {/* ---------------------------------------------------------- */}
            {/* Research evaluation note                                   */}
            {/* ---------------------------------------------------------- */}

            {result.evaluation_source && (
              <div className="mt-6 rounded-2xl bg-cyan-300/5 ring-1 ring-cyan-300/15 p-5">
                <div className="text-sm font-medium text-cyan-100">
                  About these results
                </div>

                <p className="mt-2 text-sm leading-6 text-white/55">
                  Metrics and plots are generated from the frozen Phase 7 test
                  predictions for the <strong className="text-white/80">{selectedModelInfo?.label}</strong> model.
                  Each model produces distinct visualizations based on its own
                  prediction arrays.
                </p>
              </div>
            )}

            {/* ---------------------------------------------------------- */}
            {/* Warnings                                                    */}
            {/* ---------------------------------------------------------- */}

            {(result.warnings?.length || result.warning) && (
              <div className="mt-6 space-y-3">
                {result.warning && (
                  <div className="rounded-2xl bg-yellow-300/5 ring-1 ring-yellow-300/15 p-5">
                    <div className="text-sm leading-6 text-yellow-100/70">
                      {result.warning}
                    </div>
                  </div>
                )}

                {result.warnings?.map((warning, index) => (
                  <div
                    key={`${warning}-${index}`}
                    className="rounded-2xl bg-yellow-300/5 ring-1 ring-yellow-300/15 p-5"
                  >
                    <div className="text-sm leading-6 text-yellow-100/70">
                      {warning}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* ---------------------------------------------------------- */}
            {/* Plots                                                       */}
            {/* ---------------------------------------------------------- */}

            <div className="mt-16">
              <div className={ui.kLabel}>VISUAL ANALYSIS</div>

              <h2 className={ui.h2}>Forecast diagnostics</h2>

              <p className="mt-4 max-w-3xl text-sm md:text-base leading-7 text-white/50">
                These plots summarize forecast behaviour, prediction quality,
                horizon-wise error and target-wise performance returned by the
                backend.
              </p>

              {plotEntries.length === 0 ? (
                <div className="mt-8 rounded-2xl bg-white/5 ring-1 ring-white/10 p-8">
                  <div className="text-sm font-medium text-white/75">
                    No evaluation plots are available.
                  </div>

                  <p className="mt-2 text-sm leading-6 text-white/40">
                    The forecasting request completed, but the backend did not
                    return any plotting outputs.
                  </p>
                </div>
              ) : (
                <div className="mt-10 space-y-16">
                  {PLOT_GROUPS.map((group) => {
                    const groupPlots = group.keys
                      .map((key) =>
                        plotEntries.find((entry) => entry.key === key),
                      )
                      .filter(
                        (
                          entry,
                        ): entry is {
                          plot: PlotItem;
                          key: string;
                        } => Boolean(entry),
                      );

                    if (groupPlots.length === 0) {
                      return null;
                    }

                    return (
                      <section key={group.title}>
                        <div>
                          <h3 className="text-xl md:text-2xl font-semibold text-white/90">
                            {group.title}
                          </h3>

                          <p className="mt-2 text-sm leading-6 text-white/45">
                            {group.description}
                          </p>
                        </div>

                        <div
                          className={`mt-6 grid gap-5 ${
                            groupPlots.length === 1
                              ? "grid-cols-1"
                              : "lg:grid-cols-2"
                          }`}
                        >
                          {groupPlots.map(({ plot }) => (
                            <PlotCard key={plot.name} plot={plot} />
                          ))}
                        </div>
                      </section>
                    );
                  })}

                  {/* Any future plots returned by the backend that are */}
                  {/* not part of the predefined groups are still shown. */}
                  {plotEntries.some(
                    ({ key }) =>
                      !PLOT_GROUPS.some((group) => group.keys.includes(key)),
                  ) && (
                    <section>
                      <h3 className="text-xl md:text-2xl font-semibold text-white/90">
                        Additional diagnostics
                      </h3>

                      <div className="mt-6 grid lg:grid-cols-2 gap-5">
                        {plotEntries
                          .filter(
                            ({ key }) =>
                              !PLOT_GROUPS.some((group) =>
                                group.keys.includes(key),
                              ),
                          )
                          .map(({ plot }) => (
                            <PlotCard key={plot.name} plot={plot} />
                          ))}
                      </div>
                    </section>
                  )}
                </div>
              )}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}
