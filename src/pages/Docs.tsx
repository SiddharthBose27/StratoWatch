import Navbar from "../components/Navbar";

export default function Docs() {
  return (
    <div className="px-6 md:px-12 py-14">
      <Navbar />

      <div className="mx-auto max-w-5xl">
        <div className="text-xs tracking-[0.3em] text-cyan-300/80">
          <br />
          <br />
          <br />
          DOCUMENTATION
        </div>

        <h1 className="mt-4 text-3xl md:text-5xl font-semibold">
          Developer Documentation
        </h1>

        <Section title="1. Project Overview">
          <p>
            StratoWatch is a spatio-temporal air quality forecasting framework
            for short-horizon prediction of O₃ and NO₂ across monitoring
            stations. The system combines machine learning and spatio-temporal
            modeling with a FastAPI inference backend and a React + Vite
            frontend.
          </p>

          <ul className="list-disc pl-6 mt-4 space-y-2">
            <li>Single-site XGBoost forecasting models</li>
            <li>Multi-site spatio-temporal transformer models</li>
            <li>Frozen final model artifacts for inference</li>
            <li>FastAPI endpoints for single-site and multi-site runs</li>
            <li>React + Vite frontend for model execution and visualization</li>
            <li>Forecast metrics and research-result plots</li>
          </ul>

          <p className="mt-4">
            The production-facing API uses the finalized model artifacts and
            inference pipeline rather than retraining models for each request.
          </p>
        </Section>

        <Section title="2. Installation Guide">
          <p className="mb-4">Frontend</p>

          <CodeBlock>
            {`cd /Users/siddharthbose/Desktop/Projects/stratowatch
npm install
npm run dev`}
          </CodeBlock>

          <p className="mt-8 mb-4">Backend</p>

          <CodeBlock>
            {`cd /Users/siddharthbose/Desktop/Projects/stratowatch/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`}
          </CodeBlock>
        </Section>

        <Section title="3. Requirements">
          <Table
            headers={["Component", "Version / Notes"]}
            rows={[
              ["Python", "3.12+ recommended"],
              ["Node.js", "18+"],
              ["FastAPI", "Backend API framework"],
              ["PyTorch", "Required for multi-site transformer inference"],
              ["XGBoost", "Required for single-site XGBoost inference"],
              ["NumPy / Pandas", "Data processing and numerical operations"],
              ["React / Vite", "Frontend application"],
            ]}
          />
        </Section>

        <Section title="4. Project Folder Structure">
          <CodeBlock>
            {`stratowatch/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── runners/
│   ├── stratowatch_single_site/
│   │   ├── models/
│   │   ├── outputs/
│   │   └── ...
│   ├── stratowatch_multi_site/
│   │   ├── configs/
│   │   ├── models/
│   │   ├── outputs/
│   │   └── ...
│   └── requirements.txt
├── public/
│   └── assets/
├── src/
│   ├── components/
│   └── pages/
│       ├── Home.tsx
│       ├── SingleSite.tsx
│       ├── MultiSite.tsx
│       ├── Methodology.tsx
│       └── Docs.tsx
├── package.json
├── vite.config.ts
└── README.md`}
          </CodeBlock>
        </Section>

        <Section title="5. How to Run">
          <h3 className="font-semibold mt-6">Run Backend API</h3>

          <CodeBlock>
            {`cd /Users/siddharthbose/Desktop/Projects/stratowatch/backend
source .venv/bin/activate
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`}
          </CodeBlock>

          <h3 className="font-semibold mt-6">Run Frontend</h3>

          <CodeBlock>
            {`cd /Users/siddharthbose/Desktop/Projects/stratowatch
npm run dev`}
          </CodeBlock>

          <p className="mt-4">
            Once both services are running, the frontend communicates with the
            FastAPI backend through the documented API endpoints.
          </p>
        </Section>

        <Section title="6. Model & Configuration Guide">
          <p>
            Final model artifacts and preprocessing configuration are stored
            separately for the single-site and multi-site pipelines.
          </p>

          <p className="mt-6 font-medium text-white/80">
            Multi-site configuration
          </p>

          <CodeBlock>
            {`backend/stratowatch_multi_site/configs/
- adjacency_final.npy
- target_scaler.json
- feature_scaler.json`}
          </CodeBlock>

          <p className="mt-6 font-medium text-white/80">
            Single-site final model artifacts
          </p>

          <CodeBlock>
            {`backend/stratowatch_single_site/outputs/final_baselines/xgboost/`}
          </CodeBlock>

          <p className="mt-6 font-medium text-white/80">Research outputs</p>

          <CodeBlock>
            {`backend/stratowatch_single_site/outputs/
backend/stratowatch_multi_site/outputs/`}
          </CodeBlock>

          <p className="mt-4">
            The inference API should load the finalized model and configuration
            artifacts. Training is not performed as part of a normal API
            request.
          </p>
        </Section>

        <Section title="7. API Endpoints">
          <Table
            headers={["Method", "Endpoint", "Description"]}
            rows={[
              ["GET", "/health", "Backend health check"],
              [
                "POST",
                "/api/single-site/run",
                "Run finalized single-site forecasting pipeline",
              ],
              [
                "POST",
                "/api/multi-site/run",
                "Run finalized multi-site forecasting pipeline",
              ],
            ]}
          />

          <p className="mt-4">
            The inference endpoints are intended for model execution and result
            generation. They do not retrain the underlying forecasting models.
          </p>
        </Section>

        <Section title="8. Common Errors / FAQ">
          <ul className="list-disc pl-6 space-y-3">
            <li>
              <b>Backend does not start?</b> Verify that the virtual environment
              is activated and that all dependencies in{" "}
              <code>requirements.txt</code> are installed.
            </li>

            <li>
              <b>Failed to import XGBoost?</b> Install the backend dependencies
              and verify that XGBoost is available inside the active Python
              environment.
            </li>

            <li>
              <b>PyTorch import error?</b> Verify that the installed PyTorch
              version is compatible with the active Python environment.
            </li>

            <li>
              <b>Failed to fetch?</b> Confirm that the FastAPI backend is
              running on the expected host and port before starting inference
              from the frontend.
            </li>

            <li>
              <b>Plots or result images are missing?</b> Verify that the
              expected output artifacts were generated and that the frontend
              references the correct public asset paths.
            </li>

            <li>
              <b>Inference takes longer than expected?</b> Check the backend
              terminal for model loading, inference, preprocessing, or output
              generation messages.
            </li>
          </ul>
        </Section>

        <Section title="9. Contact Us" id="contact">
          <Table
            headers={["Platform", "Link"]}
            rows={[
              ["Email", "bose.siddharth2711@gmail.com"],
              ["GitHub", "https://github.com/SiddharthBose27"],
              [
                "LinkedIn",
                "https://www.linkedin.com/in/siddharth-bose-92ab523",
              ],
            ]}
          />
        </Section>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
  id,
}: {
  title: string;
  children: React.ReactNode;
  id?: string;
}) {
  return (
    <div id={id} className="mt-14">
      <h2 className="text-xl md:text-2xl font-semibold">{title}</h2>
      <div className="mt-4 text-white/70 leading-relaxed">{children}</div>
    </div>
  );
}

function CodeBlock({ children }: { children: React.ReactNode }) {
  return (
    <pre className="mt-6 bg-black/50 ring-1 ring-white/10 rounded-2xl p-6 text-sm overflow-x-auto text-white/80 glow-hover">
      <code>{children}</code>
    </pre>
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
