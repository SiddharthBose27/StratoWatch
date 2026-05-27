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
            that predicts O₃ and NO₂ across multiple monitoring stations. The
            stack combines temporal transformers with graph-based spatial
            modeling and a FastAPI backend for running models through a web UI.
          </p>

          <ul className="list-disc pl-6 mt-4 space-y-2">
            <li>Single-site residual learning models</li>
            <li>Multi-site spatio-temporal transformer variants</li>
            <li>FastAPI inference API</li>
            <li>React + Vite frontend for uploads and visualization</li>
            <li>Metrics + plot generation</li>
          </ul>
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
pip install -r /Users/siddharthbose/Desktop/Projects/stratowatch/requirement.txt
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`}
          </CodeBlock>
        </Section>

        <Section title="3. Requirements">
          <Table
            headers={["Component", "Version / Notes"]}
            rows={[
              ["Python", "3.12+ (local uses 3.13)"] ,
              ["Node", "18+"],
              ["FastAPI", "Backend API"],
              ["PyTorch", "Required for multi-site plots"],
              ["XGBoost", "Required for single-site residual model"],
              ["NumPy / Pandas", "Data handling"],
            ]}
          />
        </Section>

        <Section title="4. Project Folder Structure">
          <CodeBlock>
{`stratowatch/
├── backend/
│   ├── app/                     # FastAPI entrypoint + runners
│   ├── stratowatch_single_site/  # Single-site pipeline
│   ├── stratowatch_multi_site/   # Multi-site pipeline
│   └── requirements.txt
├── public/
│   └── assets/plots/             # UI images
├── src/
│   ├── components/
│   └── pages/                    # Home, SingleSite, MultiSite, Methodology, Docs
├── package.json
├── vite.config.ts
└── README.md`}
          </CodeBlock>
        </Section>

        <Section title="5. How to Run">
          <h3 className="font-semibold mt-6">Run Backend API</h3>
          <CodeBlock>python3 -m uvicorn app.main:app --reload</CodeBlock>

          <h3 className="font-semibold mt-6">Run Frontend</h3>
          <CodeBlock>npm run dev</CodeBlock>
        </Section>

        <Section title="6. Configuration Guide">
          <p>Multi-site configuration files:</p>
          <CodeBlock>
{`backend/stratowatch_multi_site/configs/
- adjacency_final.npy
- target_scaler.json
- feature_scaler.json`}
          </CodeBlock>

          <p className="mt-4">Single-site plots output:</p>
          <CodeBlock>
{`backend/stratowatch_single_site/outputs/plots/
backend/stratowatch_single_site/outputs/confusion_matrices/`}
          </CodeBlock>
        </Section>

        <Section title="7. API Endpoints">
          <Table
            headers={["Method", "Endpoint", "Description"]}
            rows={[
              ["POST", "/api/single-site/run", "Run single-site inference"],
              ["POST", "/api/multi-site/run", "Run multi-site inference"],
              ["GET", "/health", "Health check"],
            ]}
          />
        </Section>

        <Section title="8. Common Errors / FAQ">
          <ul className="list-disc pl-6 space-y-2">
            <li>
              <b>Model keeps running?</b> Check backend logs for long-running
              plot generation and ensure torch is installed.
            </li>
            <li>
              <b>Failed to import xgboost?</b> Install via `pip install xgboost`.
            </li>
            <li>
              <b>Plots not showing?</b> Verify images are in `public/assets/plots`.
            </li>
          </ul>
        </Section>

        <Section title="9. Contact Us" id="contact">
          <Table
            headers={["Platform", "Link"]}
            rows={[
              ["Email", "bose.siddharth2711@gmail.com"],
              ["GitHub", "https://github.com/SiddharthBose27"],
              ["LinkedIn", "https://www.linkedin.com/in/siddharth-bose-92ab523"],
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

function Table({
  headers,
  rows,
}: {
  headers: string[];
  rows: string[][];
}) {
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
