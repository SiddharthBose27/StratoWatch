import Navbar from "../components/Navbar";
import { ui } from "../styles/ui";

const SINGLE_SITE_RESULTS = [
  {
    model: "XGBoost",
    mae: "0.4039",
    rmse: "0.5906",
    r2: "0.3848",
    note: "Best overall single-site result",
  },
  {
    model: "Random Forest",
    mae: "0.4392",
    rmse: "0.6526",
    r2: "0.2489",
    note: "Second-best tree baseline",
  },
  {
    model: "TCN",
    mae: "0.5588",
    rmse: "0.7427",
    r2: "0.0273",
    note: "Temporal convolutional model",
  },
  {
    model: "Transformer",
    mae: "0.5700",
    rmse: "0.7524",
    r2: "0.0017",
    note: "Temporal attention model",
  },
  {
    model: "LSTM",
    mae: "0.6616",
    rmse: "0.8296",
    r2: "-0.2138",
    note: "Recurrent temporal baseline",
  },
];

const MULTI_SITE_RESULTS = [
  {
    model: "ST Transformer",
    mae: "18.8349",
    rmse: "27.9747",
    r2: "0.3437",
    note: "Best MAE",
  },
  {
    model: "Static Graph-ST",
    mae: "19.0384",
    rmse: "27.0781",
    r2: "0.3850",
    note: "Best R² among the listed models",
  },
  {
    model: "Dynamic Wind Graph-ST",
    mae: "19.0465",
    rmse: "26.9455",
    r2: "0.3911",
    note: "Best RMSE and R²",
  },
];

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-5">
      <div className="text-xs tracking-[0.2em] uppercase text-white/40">
        {label}
      </div>

      <div className="mt-2 text-xl font-semibold text-white">{value}</div>
    </div>
  );
}

function Section({
  label,
  title,
  children,
}: {
  label: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className={`${ui.container} ${ui.section}`}>
      <div className={ui.kLabel}>{label}</div>

      <h2 className={ui.h2}>{title}</h2>

      <div className="mt-8">{children}</div>
    </section>
  );
}

export default function Methodology() {
  return (
    <main className={ui.page}>
      <Navbar />

      {/* ================================================================= */}
      {/* HEADER                                                            */}
      {/* ================================================================= */}

      <section className={`${ui.container} pt-20 md:pt-28`}>
        <div className="max-w-4xl">
          <div className={ui.kLabel}>METHODOLOGY</div>

          <h1 className="mt-4 text-4xl md:text-6xl font-semibold tracking-tight">
            How StratoWatch forecasts air quality
          </h1>

          <p className="mt-6 text-base md:text-lg leading-8 text-white/60">
            StratoWatch evaluates short-horizon forecasts for ground-level ozone
            (O3) and nitrogen dioxide (NO2) using temporal and spatio-temporal
            machine-learning models.
          </p>
        </div>
      </section>

      {/* ================================================================= */}
      {/* PROBLEM                                                           */}
      {/* ================================================================= */}

      <Section label="01 — PROBLEM" title="Forecasting setup">
        <div className="grid lg:grid-cols-2 gap-6">
          <div className={ui.card}>
            <div className={ui.kLabel}>INPUT</div>

            <h3 className="mt-4 text-2xl font-semibold">
              Historical observations and atmospheric features
            </h3>

            <p className="mt-4 text-sm leading-7 text-white/55">
              The single-site pipeline uses a 24-hour historical input window
              containing 134 features. Tree-based models receive the flattened
              representation, while sequence models preserve the temporal
              structure.
            </p>
          </div>

          <div className={ui.card}>
            <div className={ui.kLabel}>OUTPUT</div>

            <h3 className="mt-4 text-2xl font-semibold">
              Six-hour multi-target forecast
            </h3>

            <p className="mt-4 text-sm leading-7 text-white/55">
              Each final single-site model produces six future forecast horizons
              for two targets: O3 and NO2. The resulting prediction tensor has
              the form 6 × 2 per sample.
            </p>
          </div>
        </div>
      </Section>

      {/* ================================================================= */}
      {/* SINGLE SITE MODELS                                                */}
      {/* ================================================================= */}

      <Section label="02 — SINGLE SITE" title="Final model comparison">
        <div className="overflow-x-auto rounded-3xl bg-white/[0.035] ring-1 ring-white/10">
          <table className="w-full min-w-[760px] text-left">
            <thead>
              <tr className="border-b border-white/10">
                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  Model
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  MAE
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  RMSE
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  R²
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  Note
                </th>
              </tr>
            </thead>

            <tbody>
              {SINGLE_SITE_RESULTS.map((result, index) => (
                <tr
                  key={result.model}
                  className="border-b border-white/5 last:border-b-0"
                >
                  <td className="px-6 py-5 font-medium text-white/85">
                    <div className="flex items-center gap-3">
                      {index === 0 && (
                        <span className="rounded-full bg-cyan-300/10 px-2.5 py-1 text-[10px] tracking-[0.15em] uppercase text-cyan-200 ring-1 ring-cyan-300/20">
                          Best
                        </span>
                      )}

                      {result.model}
                    </div>
                  </td>

                  <td className="px-6 py-5 font-mono text-sm text-white/70">
                    {result.mae}
                  </td>

                  <td className="px-6 py-5 font-mono text-sm text-white/70">
                    {result.rmse}
                  </td>

                  <td className="px-6 py-5 font-mono text-sm text-white/70">
                    {result.r2}
                  </td>

                  <td className="px-6 py-5 text-sm text-white/45">
                    {result.note}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-8 grid sm:grid-cols-3 gap-4">
          <MetricBox label="Best MAE" value="0.4039" />

          <MetricBox label="Best RMSE" value="0.5906" />

          <MetricBox label="Best R²" value="0.3848" />
        </div>

        <div className="mt-8 rounded-2xl bg-cyan-300/5 ring-1 ring-cyan-300/15 p-6">
          <div className="text-sm font-medium text-cyan-100">
            Current final-model formulation
          </div>

          <p className="mt-2 text-sm leading-7 text-white/55">
            The final single-site models use direct target prediction. The
            current evaluation should therefore not be described as the old
            residual-learning pipeline.
          </p>
        </div>
      </Section>

      {/* ================================================================= */}
      {/* MODEL DETAILS                                                     */}
      {/* ================================================================= */}

      <Section label="03 — MODELS" title="What each model contributes">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          <div className={ui.subtle}>
            <div className={ui.kLabel}>TREE MODEL</div>

            <h3 className="mt-3 text-xl font-semibold">XGBoost</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              Gradient-boosted decision trees operating on the flattened 24-hour
              feature window. It provides the strongest final single-site
              benchmark.
            </p>
          </div>

          <div className={ui.subtle}>
            <div className={ui.kLabel}>TREE MODEL</div>

            <h3 className="mt-3 text-xl font-semibold">Random Forest</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              An ensemble of decision trees used as a second non-neural baseline
              for comparison against sequence models.
            </p>
          </div>

          <div className={ui.subtle}>
            <div className={ui.kLabel}>RECURRENT</div>

            <h3 className="mt-3 text-xl font-semibold">LSTM</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              A recurrent architecture that models temporal dependencies
              directly from the ordered input sequence.
            </p>
          </div>

          <div className={ui.subtle}>
            <div className={ui.kLabel}>CONVOLUTIONAL</div>

            <h3 className="mt-3 text-xl font-semibold">TCN</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              Temporal convolutional modelling for learning patterns across the
              historical sequence.
            </p>
          </div>

          <div className={ui.subtle}>
            <div className={ui.kLabel}>ATTENTION</div>

            <h3 className="mt-3 text-xl font-semibold">Transformer</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              Temporal self-attention is used to model relationships across the
              historical input window.
            </p>
          </div>

          <div className={ui.subtle}>
            <div className={ui.kLabel}>COMMON OUTPUT</div>

            <h3 className="mt-3 text-xl font-semibold">6 × 2 forecast</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              All five final single-site models are evaluated against the same
              six-horizon, two-target forecasting objective.
            </p>
          </div>
        </div>
      </Section>

      {/* ================================================================= */}
      {/* EVALUATION                                                        */}
      {/* ================================================================= */}

      <Section label="04 — EVALUATION" title="Frozen test evaluation">
        <div className="grid lg:grid-cols-3 gap-5">
          <div className={ui.subtle}>
            <div className={ui.kLabel}>SPLIT</div>

            <h3 className="mt-3 text-xl font-semibold">Phase 7 test set</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              Final metrics are produced against the official frozen
              chronological test artifact.
            </p>
          </div>

          <div className={ui.subtle}>
            <div className={ui.kLabel}>METRICS</div>

            <h3 className="mt-3 text-xl font-semibold">MAE · RMSE · R²</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              Performance is reported overall, by target and across the six
              forecast horizons.
            </p>
          </div>

          <div className={ui.subtle}>
            <div className={ui.kLabel}>REPRODUCIBILITY</div>

            <h3 className="mt-3 text-xl font-semibold">Frozen inference</h3>

            <p className="mt-3 text-sm leading-6 text-white/50">
              The application evaluates saved final model artifacts rather than
              retraining models during a request.
            </p>
          </div>
        </div>
      </Section>

      {/* ================================================================= */}
      {/* VISUAL DIAGNOSTICS                                                */}
      {/* ================================================================= */}

      <Section label="05 — DIAGNOSTICS" title="Evaluation visualizations">
        <div className="grid md:grid-cols-2 gap-5">
          {[
            {
              title: "Actual vs predicted",
              text: "Shows how forecast trajectories compare with observed target values.",
            },
            {
              title: "Prediction scatter",
              text: "Shows the relationship between predicted and observed values.",
            },
            {
              title: "Residual distribution",
              text: "Shows the distribution of prediction errors for each target.",
            },
            {
              title: "Horizon performance",
              text: "Shows how MAE and RMSE change across the six forecast horizons.",
            },
            {
              title: "Target error by horizon",
              text: "Separates O3 and NO2 error behaviour across future horizons.",
            },
            {
              title: "Target comparison",
              text: "Compares final performance between the two forecast targets.",
            },
          ].map((item) => (
            <div key={item.title} className={ui.subtle}>
              <h3 className="text-lg font-semibold">{item.title}</h3>

              <p className="mt-2 text-sm leading-6 text-white/50">
                {item.text}
              </p>
            </div>
          ))}
        </div>
      </Section>

      {/* ================================================================= */}
      {/* MULTI SITE                                                        */}
      {/* ================================================================= */}

      <Section label="06 — MULTI SITE" title="Spatio-temporal modelling">
        <div className="max-w-4xl">
          <p className="text-sm md:text-base leading-7 text-white/55">
            The multi-site pipeline extends forecasting from a single station to
            a graph of monitoring locations. Spatial relationships are
            represented explicitly so that temporal information can be combined
            with interactions between sites.
          </p>
        </div>

        <div className="mt-8 overflow-x-auto rounded-3xl bg-white/[0.035] ring-1 ring-white/10">
          <table className="w-full min-w-[760px] text-left">
            <thead>
              <tr className="border-b border-white/10">
                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  Model
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  MAE
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  RMSE
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  R²
                </th>

                <th className="px-6 py-5 text-xs tracking-[0.2em] uppercase text-white/40">
                  Note
                </th>
              </tr>
            </thead>

            <tbody>
              {MULTI_SITE_RESULTS.map((result) => (
                <tr
                  key={result.model}
                  className="border-b border-white/5 last:border-b-0"
                >
                  <td className="px-6 py-5 font-medium text-white/85">
                    {result.model}
                  </td>

                  <td className="px-6 py-5 font-mono text-sm text-white/70">
                    {result.mae}
                  </td>

                  <td className="px-6 py-5 font-mono text-sm text-white/70">
                    {result.rmse}
                  </td>

                  <td className="px-6 py-5 font-mono text-sm text-white/70">
                    {result.r2}
                  </td>

                  <td className="px-6 py-5 text-sm text-white/45">
                    {result.note}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-8 rounded-2xl bg-yellow-300/5 ring-1 ring-yellow-300/15 p-6">
          <div className="text-sm font-medium text-yellow-100">
            Seven-site research configuration
          </div>

          <p className="mt-2 text-sm leading-7 text-white/55">
            The trained multi-site research checkpoint corresponds to the
            seven-site configuration. The current application accepts other site
            counts for interface compatibility, but those configurations do not
            represent newly trained research checkpoints.
          </p>
        </div>
      </Section>

      {/* ================================================================= */}
      {/* DATASET BEHAVIOUR                                                 */}
      {/* ================================================================= */}

      <Section
        label="07 — DATA HANDLING"
        title="Why uploaded datasets can return the same metrics"
      >
        <div className="max-w-4xl rounded-3xl bg-white/[0.035] ring-1 ring-white/10 p-8">
          <p className="text-sm md:text-base leading-7 text-white/55">
            The current application accepts CSV/XLSX files as compatibility
            inputs, but the official research evaluation uses the frozen Phase 7
            test artifact. Consequently, changing the uploaded file does not
            currently replace the official research test set.
          </p>

          <p className="mt-5 text-sm md:text-base leading-7 text-white/55">
            This is intentional: the reported research metrics remain
            reproducible and consistent with the locked evaluation results
            rather than changing according to arbitrary uploaded data.
          </p>
        </div>
      </Section>

      {/* ================================================================= */}
      {/* SUMMARY                                                           */}
      {/* ================================================================= */}

      <section className={`${ui.container} pb-24 md:pb-32`}>
        <div className="rounded-3xl bg-white/[0.035] ring-1 ring-white/10 p-8 md:p-10">
          <div className={ui.kLabel}>FINAL PIPELINE</div>

          <div className="mt-6 flex flex-wrap items-center gap-3 text-sm">
            {[
              "24-hour history",
              "134 features",
              "Frozen model",
              "6 horizons",
              "O3 + NO2",
              "MAE / RMSE / R²",
              "Diagnostic plots",
            ].map((item, index, items) => (
              <div key={item} className="flex items-center gap-3">
                <span className="rounded-full bg-white/5 ring-1 ring-white/10 px-4 py-2 text-white/65">
                  {item}
                </span>

                {index < items.length - 1 && (
                  <span className="text-white/20">→</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
