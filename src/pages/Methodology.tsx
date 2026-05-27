import Navbar from "../components/Navbar";

export default function Methodology() {
  return (
    <div className="px-6 md:px-12 py-14">
      <Navbar />

      <div className="mx-auto max-w-5xl">
        <div className="text-xs tracking-[0.3em] text-cyan-300/80">
          <br />
          <br />
          <br />
          METHODOLOGY
        </div>

        <h1 className="mt-4 text-3xl md:text-5xl font-semibold">
          Research Methodology
        </h1>

        <Section title="1. Problem Statement">
          <p>
            The objective of this study is to develop a robust multi-site
            spatio-temporal forecasting framework for air quality prediction,
            specifically targeting O₃ (Ozone) and NO₂ (Nitrogen Dioxide)
            concentrations. The system models temporal dependencies within each
            monitoring station while capturing spatial correlations across
            geographically distributed sites.
          </p>
        </Section>

        <Section title="2. System Overview">
          <p>
            The system follows a structured machine learning pipeline from raw
            data ingestion to deployment-ready inference.
          </p>

          <ImageCard
            title="End-to-end pipeline overview"
            src="/assets/plots/loss_curve.png"
          />
        </Section>

        <Section title="3. Dataset Description">
          <Table
            headers={["Category", "Details"]}
            rows={[
              ["Monitoring Sites", "7 air quality monitoring stations"],
              ["Targets", "O₃ concentration, NO₂ concentration"],
              ["Input Window (Tin)", "24 hours historical data"],
              ["Forecast Horizon (Tout)", "6 hours ahead"],
              ["Features", "Meteorological + pollutant features"],
              ["Temporal Resolution", "Hourly"],
              ["Dataset Type", "Multivariate time-series"],
            ]}
          />
        </Section>

        <Section title="4. Data Preprocessing">
          <ul className="list-disc pl-6 space-y-2">
            <li>
              Missing values handled using masking strategy with zero
              imputation.
            </li>
            <li>Global feature scaling applied using standard normalization.</li>
            <li>Target scaling stored and inverse-transformed during evaluation.</li>
            <li>Train / Validation / Test split using sliding window generation.</li>
            <li>Union timeline alignment across all monitoring stations.</li>
          </ul>
        </Section>

        <Section title="5. Model Architecture">
          <p>
            The core architecture is a Graph-Enhanced Spatio-Temporal
            Transformer integrating temporal attention and spatial graph
            propagation.
          </p>

          <Table
            headers={["Component", "Description"]}
            rows={[
              ["Input Projection", "Linear projection to d_model = 128"],
              ["Temporal Encoder", "Multi-head self-attention (per-site)"],
              ["Graph Layer", "Static / Dynamic wind-weighted adjacency"],
              ["Spatial Encoder", "Attention across monitoring sites"],
              ["Output Head", "Multi-horizon regression (6-step ahead)"],
              ["Loss Function", "SmoothL1Loss"],
              ["Optimizer", "Adam"],
              ["Dropout", "0.1"],
            ]}
          />
        </Section>

        <Section title="6. Training Strategy">
          <Table
            headers={["Parameter", "Value"]}
            rows={[
              ["Epochs", "Early stopping enabled"],
              ["Batch Size", "64"],
              ["Learning Rate", "1e-3"],
              ["Device", "CPU / MPS (Apple Silicon)"],
              ["Regularization", "Dropout 0.1"],
            ]}
          />
        </Section>

        <Section title="7. Evaluation Metrics">
          <ul className="list-disc pl-6 space-y-2">
            <li>Mean Absolute Error (MAE)</li>
            <li>Root Mean Squared Error (RMSE)</li>
            <li>R² Score (where applicable)</li>
            <li>Horizon-wise MAE analysis</li>
            <li>Per-site performance evaluation</li>
          </ul>
        </Section>

        <Section title="8. Results">
          <div className="grid gap-6 md:grid-cols-2">
            <ImageCard
              title="Horizon-wise MAE"
              src="/assets/plots/o3_true_vs_pred.png"
            />
            <ImageCard
              title="Residual Distribution"
              src="/assets/plots/o3_residual_distribution.png"
            />
            <ImageCard
              title="Confusion Matrix (Severity Bins)"
              src="/assets/plots/XGB_O3_cm.png"
            />
            <ImageCard
              title="Site Embedding PCA"
              src="/assets/plots/o3_scatter.png"
            />
          </div>
        </Section>

        <Section title="9. Baseline Comparison">
          <Table
            headers={["Model", "MAE", "RMSE"]}
            rows={[
              ["ST Transformer", "18.69", "26.66"],
              ["Static Graph-ST", "19.01", "26.99"],
              ["Dynamic Wind Graph-ST", "19.00", "26.99"],
            ]}
          />
        </Section>

        <Section title="10. Limitations & Future Work">
          <ul className="list-disc pl-6 space-y-2">
            <li>
              Limited improvement observed from wind-aware adjacency
              integration.
            </li>
            <li>
              Underestimation of extreme pollution spikes remains a challenge.
            </li>
            <li>
              Future work includes adaptive graph learning and uncertainty
              estimation.
            </li>
            <li>
              Expansion to additional pollutants and larger geographic regions.
            </li>
          </ul>
        </Section>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-14">
      <h2 className="text-xl md:text-2xl font-semibold">{title}</h2>
      <div className="mt-4 text-white/70 leading-relaxed">{children}</div>
    </div>
  );
}

function Table({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="mt-6 overflow-x-auto">
      <table className="w-full text-sm text-left border-collapse">
        <thead>
          <tr className="border-b border-white/10">
            {headers.map((h) => (
              <th key={h} className="py-3 pr-6 text-white/80 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-white/5">
              {row.map((cell, j) => (
                <td key={j} className="py-3 pr-6 text-white/60">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ImageCard({ title, src }: { title: string; src: string }) {
  return (
    <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-5 glow-hover">
      <div className="text-white/70 text-sm mb-3">{title}</div>
      <img
        src={src}
        alt={title}
        className="w-full rounded-xl ring-1 ring-white/10"
      />
    </div>
  );
}
