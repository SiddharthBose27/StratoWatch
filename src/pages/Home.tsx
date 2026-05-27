import hero from "../assets/hero.jpg";
import { motion, type Variants } from "framer-motion";
import { Link } from "react-router-dom";


const premiumCard =
  "relative overflow-hidden glow-hover rounded-2xl bg-white/5 ring-1 ring-white/10 p-7 " +
  "transition-all duration-300 hover:-translate-y-1 hover:ring-cyan-300/30 " +
  "hover:shadow-[0_18px_60px_rgba(0,0,0,0.65)] " +
  "before:absolute before:inset-0 before:opacity-0 before:transition before:duration-300 " +
  "hover:before:opacity-100 " +
  "before:bg-[radial-gradient(circle_at_30%_30%,rgba(34,211,238,0.18),transparent_60%)]";


const fadeUp: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.08 * i, duration: 0.6, ease: "easeOut" },
  }),
};

export default function Home() {
  return (
    <div className="bg-black text-white">
      {/* ============ HERO SECTION ============ */}
      <section id="home" className="relative min-h-screen overflow-hidden">
        {/* Background image */}
        <div className="absolute inset-0">
          <img
            src={hero}
            className="w-full h-full object-cover"
            alt="StratoWatch hero"
          />
        </div>

        {/* Overlays */}
        <div className="absolute inset-0 bg-linear-gradient-to-b from-black/50 via-black/70 to-black/95" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_40%,rgba(56,189,248,0.20),transparent_55%)]" />

        {/* Content */}
        <div className="relative z-10 min-h-screen px-6 md:px-12">
          {/* Navbar */}
          <header className="flex items-center justify-between py-8">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-white/10 ring-1 ring-white/15">
                <svg
                  className="h-5 w-5 text-cyan-300 animate-satellite satellite-glow"
                  viewBox="0 0 32 32"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  aria-hidden="true"
                >
                  <rect
                    x="12"
                    y="12"
                    width="8"
                    height="8"
                    rx="1.6"
                    fill="currentColor"
                  />
                  <rect
                    x="2"
                    y="11"
                    width="7"
                    height="10"
                    rx="1.2"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                  <rect
                    x="23"
                    y="11"
                    width="7"
                    height="10"
                    rx="1.2"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                  <line
                    x1="9"
                    y1="16"
                    x2="12"
                    y2="16"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                  <line
                    x1="20"
                    y1="16"
                    x2="23"
                    y2="16"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                  <circle cx="16" cy="7" r="2" fill="currentColor" />
                  <line
                    x1="16"
                    y1="9.5"
                    x2="16"
                    y2="12"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                  <path
                    d="M22.5 22.5l4 4"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                  />
                  <circle
                    cx="27"
                    cy="27"
                    r="2.4"
                    stroke="currentColor"
                    strokeWidth="1.4"
                  />
                </svg>
              </div>
              <span className="tracking-[0.25em] text-white/80 text-sm">
                STRATOWATCH
              </span>
            </div>

            {/* Navigation */}
            <nav className="hidden md:flex items-center gap-8 text-sm text-white/70">
              <Link to="/" className="hover:text-white transition nav-link">
                Home
              </Link>

              <Link
                to="/single-site"
                className="hover:text-white transition nav-link"
              >
                Single-Site
              </Link>

              <Link
                to="/multi-site"
                className="hover:text-white transition nav-link"
              >
                Multi-Site
              </Link>

              <Link
                to="/methodology"
                className="hover:text-white transition nav-link"
              >
                Methodology
              </Link>

              <Link to="/docs" className="hover:text-white transition nav-link">
                Docs
              </Link>
            </nav>

            {/* CTA */}
            <Link
              to="/docs#contact"
              className="rounded-xl bg-white/10 px-4 py-2 text-sm text-white/80 ring-1 ring-white/15 hover:bg-white/15 hover:text-white transition"
            >
              Get in touch
            </Link>
          </header>

          {/* Hero body */}
          <div className="grid min-h-[calc(100vh-120px)] items-center">
            <div className="max-w-3xl">
              <motion.div
                initial="hidden"
                animate="visible"
                variants={fadeUp}
                custom={0}
                className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs text-white/70 ring-1 ring-white/15"
              >
                <span className="h-1.5 w-1.5 rounded-full bg-cyan-300" />
                Atmospheric Intelligence Platform
              </motion.div>

              <motion.h1
                initial="hidden"
                animate="visible"
                variants={fadeUp}
                custom={1}
                className="mt-6 text-5xl md:text-6xl font-extrabold tracking-[0.14em]"
              >
                STRATOWATCH
              </motion.h1>

              <motion.p
                initial="hidden"
                animate="visible"
                variants={fadeUp}
                custom={2}
                className="mt-6 text-white/70 leading-relaxed text-lg max-w-2xl"
              >
                A forecasting + evaluation hub for air-quality intelligence:
                single-site residual correction and multi-site spatio-temporal
                learning for O₃ and NO₂ — with metrics, plots, and dashboards.
              </motion.p>

              <motion.div
                initial="hidden"
                animate="visible"
                variants={fadeUp}
                custom={3}
                className="mt-8 flex flex-wrap items-center gap-4"
              >
                <Link
                  to="/single-site"
                  className="rounded-xl bg-cyan-400 px-6 py-3 text-black font-semibold hover:bg-cyan-300 transition"
                >
                  Single Site
                </Link>
                <Link
                  to="/multi-site"
                  className="rounded-xl bg-cyan-400 px-6 py-3 text-black font-semibold hover:bg-cyan-300 transition"
                >
                  Multiple Site
                </Link>
              </motion.div>

              {/* Mini highlight chips */}
              <motion.div
                initial="hidden"
                animate="visible"
                variants={fadeUp}
                custom={4}
                className="mt-10 flex flex-wrap gap-8 text-sm"
              >
                <div>
                  <div className="text-white font-semibold">Single + Multi</div>
                  <div className="text-white/50">Forecast systems</div>
                </div>
                <div>
                  <div className="text-white font-semibold">
                    Residual + Graph
                  </div>
                  <div className="text-white/50">Learning styles</div>
                </div>
                <div>
                  <div className="text-white font-semibold">O₃ + NO₂</div>
                  <div className="text-white/50">Targets</div>
                </div>
              </motion.div>

              {/* Scroll hint */}
              <div className="mt-14 flex items-center gap-3 text-white/50 text-xs">
                <span className="inline-block h-6 w-px bg-white/20" />
                Scroll to explore
              </div>
            </div>
          </div>
        </div>

        {/* bottom fade to make next section blend */}
        <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-28 bg-linear-gradient-to-b from-transparent to-black" />
      </section>

      {/* ============ PLATFORM OVERVIEW ============ */}
      <section id="overview" className="relative py-20 px-6 md:px-12">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
            variants={fadeUp}
            custom={0}
            className="text-xs tracking-[0.3em] text-cyan-300/80"
          >
            PLATFORM OVERVIEW
          </motion.div>

          <motion.h2
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
            variants={fadeUp}
            custom={1}
            className="mt-4 text-3xl md:text-4xl font-semibold tracking-tight"
          >
            What is StratoWatch?
          </motion.h2>

          <div className="mt-10 grid gap-8 md:grid-cols-2">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.2 }}
              variants={fadeUp}
              custom={2}
              className={premiumCard}
            >
              <div className="text-white/80 font-medium">Definition</div>
              <p className="mt-3 text-white/60 leading-relaxed">
                StratoWatch is an atmospheric intelligence platform that
                connects forecasting models, evaluation metrics, and dashboards
                into a single pipeline. It covers two tracks: (1) single-site
                residual correction and (2) multi-site spatio-temporal
                forecasting across stations for O₃ and NO₂.
              </p>
            </motion.div>

            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.2 }}
              variants={fadeUp}
              custom={3}
              className={premiumCard}
            >
              <div className="text-white/80 font-medium">Why it matters</div>
              <p className="mt-3 text-white/60 leading-relaxed">
                Forecasts often miss spikes and local dynamics. Residual
                learning models the correction term (y = forecast + Δ).
                Multi-site modeling captures spatial + temporal dependencies so
                station networks behave like a system — not isolated sensors.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ============ TWO CORE SYSTEMS (SPLIT PANEL) ============ */}
      <section className="py-20 px-6 md:px-12">
        <div className="max-w-6xl mx-auto">
          <div className="text-xs tracking-[0.3em] text-cyan-300/80">
            TWO CORE SYSTEMS
          </div>

          <div className="mt-8 grid gap-8 lg:grid-cols-2">
            {/* Single-site */}
            <div id="single" className={`${premiumCard} p-8`}>
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.18),transparent_60%)]" />
              <div className="relative">
                <div className="text-white font-semibold text-xl">
                  Single-Site Residual Forecasting
                </div>
                <p className="mt-3 text-white/60 leading-relaxed">
                  One station. You start with a forecast baseline and train
                  ML/DL models to predict the residual correction (Δ). This
                  improves real-world accuracy for station-level monitoring.
                </p>

                <div className="mt-6 text-white/70 text-sm font-medium">
                  Models included
                </div>
                <ul className="mt-3 space-y-2 text-white/60 text-sm">
                  <li>• LSTM</li>
                  <li>• XGBoost (best)</li>
                  <li>• TCN </li>
                  <li>• Transformer</li>
                </ul>

                <div className="mt-8 flex gap-3"></div>
              </div>
            </div>

            {/* Multi-site */}
            <div id="multi" className={`${premiumCard} p-8`}>
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(34,211,238,0.16),transparent_60%)]" />
              <div className="relative">
                <div className="text-white font-semibold text-xl">
                  Multi-Site Spatio-Temporal Forecasting
                </div>
                <p className="mt-3 text-white/60 leading-relaxed">
                  Multiple stations (7). Uses 24h context to predict 6h horizon
                  for O₃ and NO₂, capturing spatial + temporal dependencies.
                </p>

                <div className="mt-6 text-white/70 text-sm font-medium">
                  Variants tested
                </div>
                <ul className="mt-3 space-y-2 text-white/60 text-sm">
                  <li>• ST Transformer (no graph)</li>
                  <li>• Static Graph-ST (fixed adjacency)</li>
                  <li>• Dynamic Wind Graph-ST (wind-weighted adjacency)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ PERFORMANCE SNAPSHOT ============ */}
      <section id="performance" className="py-20 px-6 md:px-12">
        <div className="max-w-6xl mx-auto">
          <div className="text-xs tracking-[0.3em] text-cyan-300/80">
            PERFORMANCE SNAPSHOT
          </div>

          <h2 className="mt-4 text-3xl md:text-4xl font-semibold">
            Verified MAE / RMSE / R² highlights
          </h2>

          <p className="mt-3 text-white/55 max-w-3xl leading-relaxed">
            Metrics are computed on a held-out test set with no data leakage.
          </p>

          {/* ====== CARDS GRID ====== */}
          <div className="mt-10 grid gap-8 lg:grid-cols-2 items-stretch">
            {/* ================== SINGLE SITE ================== */}
            <div className="premium-card p-9 h-full">
              <div className="flex items-start justify-between gap-6">
                <div>
                  <div className="text-white font-semibold text-xl">
                    Single-Site
                  </div>
                  <div className="mt-1 text-white/60 text-sm">
                    Best model:{" "}
                    <span className="text-white/80">
                      XGBoost — Residual Learning
                    </span>
                  </div>
                </div>
                <div className="text-[11px] tracking-[0.28em] text-white/45">
                  VERIFIED
                </div>
              </div>

              {/* BEST MODEL METRICS - CENTERED LIKE MULTI-SITE */}
              <div className="mt-7 flex justify-center">
                <div className="grid grid-cols-3 gap-6 max-w-2xl w-full">
                  {[
                    { k: "MAE", v: "17.95" },
                    { k: "RMSE", v: "26.69" },
                    { k: "R²", v: "0.42" },
                  ].map((m) => (
                    <div
                      key={m.k}
                      className="metric-tile glow-hover p-6 text-center flex flex-col items-center justify-center"
                    >
                      <div className="text-white/55 text-xs tracking-[0.22em]">
                        {m.k}
                      </div>
                      <div className="mt-3 text-4xl font-semibold">{m.v}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* MODELS EVALUATED */}
              <div className="mt-8">
                <div className="text-white/60 text-sm">Models evaluated</div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    "LSTM (R² 0.184)",
                    "TCN (R² 0.2938)",
                    "Forecast-only baseline (R² -0.21)",
                    "Transformer (Temporal Encoder) (R² ~0.26–0.33)",
                    "XGBoost Residual (R² 0.4208) — Best",
                  ].map((t) => (
                    <span
                      key={t}
                      className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/60 ring-1 ring-white/10
                           hover:bg-white/8 hover:text-white/75 transition"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>

              <p className="mt-8 text-white/50 text-sm leading-relaxed">
                XGBoost with residual learning improved station-level accuracy
                versus a forecast-only baseline. Neural baselines (LSTM/TCN)
                provide strong temporal references.
              </p>

              <div className="mt-6 flex flex-wrap gap-3"></div>
            </div>

            {/* ================== MULTI SITE ================== */}
            <div className="premium-card p-9 h-full">
              <div className="flex items-start justify-between gap-6">
                <div>
                  <div className="text-white font-semibold text-xl">
                    Multi-Site (Aggregate)
                  </div>
                  <div className="mt-1 text-white/60 text-sm">
                    Best model:{" "}
                    <span className="text-white/80">ST Transformer</span>
                  </div>
                  <div className="mt-1 text-white/45 text-xs">
                    7 sites • 24h context → 6h horizon • O₃ + NO₂
                  </div>
                </div>

                <div className="text-[11px] tracking-[0.28em] text-white/45">
                  VERIFIED
                </div>
              </div>

              {/* BEST MODEL METRICS */}
              <div className="mt-7 flex justify-center">
                <div className="grid grid-cols-2 gap-6 max-w-md w-full">
                  {[
                    { k: "MAE", v: "18.69" },
                    { k: "RMSE", v: "26.66" },
                  ].map((m) => (
                    <div
                      key={m.k}
                      className="metric-tile glow-hover p-6 text-center flex flex-col items-center justify-center"
                    >
                      <div className="text-white/55 text-xs tracking-[0.22em]">
                        {m.k}
                      </div>
                      <div className="mt-3 text-4xl font-semibold">{m.v}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* MODELS EVALUATED */}
              <div className="mt-8">
                <div className="text-white/60 text-sm">Models evaluated</div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    "Static Graph-ST (MAE 19.01, RMSE 26.99)",
                    "Dynamic Wind Graph-ST (MAE 19.0047, RMSE 26.99)",
                  ].map((t) => (
                    <span
                      key={t}
                      className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/60 ring-1 ring-white/10
                           hover:bg-white/8 hover:text-white/75 transition"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>

              <p className="mt-8 text-white/50 text-sm leading-relaxed">
                Multi-site forecasting learns temporal patterns plus spatial
                influence across stations. Graph variants inject adjacency
                structure; wind-weighted graphs model dynamic cross-site
                interactions for better robustness under missing/noisy sensors.
              </p>

              <div className="mt-6 flex flex-wrap gap-3"></div>
            </div>
          </div>

          {/* helper note */}
          <p className="mt-8 text-white/45 text-sm">
            Tip: if you later compute Multi-Site R², just replace the “—” tile
            value. If you’d rather hide R² entirely for Multi-Site, change the
            grid to
            <span className="text-white/70"> grid-cols-2</span> and remove the
            third tile.
          </p>
        </div>
      </section>

      {/* ============ O2/NO2 MONITORING ============ */}
      <section id="monitoring" className="py-20 px-6 md:px-12">
        <div className="max-w-6xl mx-auto">
          <div className="text-xs tracking-[0.32em] text-cyan-300/80">
            MONITORING & FORECASTING
          </div>
          <h2 className="md:text-5xl font-semibold tracking-[0.08em]">
            O₂ and NO₂ Monitoring & Forecasting
          </h2>
          <div className="mt-4 h-0.5 w-16 rounded-full bg-cyan-300/80" />

          <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="relative rounded-3xl backdrop-blur-md bg-gradient-to-br from-cyan-500/10 via-transparent to-blue-500/10 border border-white/10 shadow-[inset_0_0_24px_rgba(255,255,255,0.03)] p-8 transition-all duration-300 ease-in-out hover:-translate-y-2 hover:shadow-[0_0_40px_rgba(0,255,255,0.2)]">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,255,255,0.8)]" />
                <h3 className="text-2xl md:text-3xl font-semibold tracking-wide">
                  O₂ [ Oxygen ]
                </h3>
              </div>
              <div className="mt-3 h-0.5 w-12 rounded-full bg-cyan-300/70" />
              <p className="mt-5 text-white/60 leading-relaxed max-w-md">
                Residual learning refines oxygen forecasts by modeling the
                correction term (forecast + Δ), while spatio-temporal signals
                help reduce bias in station-level predictions.
              </p>
            </div>

            <div className="relative rounded-3xl backdrop-blur-md bg-gradient-to-br from-cyan-500/10 via-transparent to-blue-500/10 border border-white/10 shadow-[inset_0_0_24px_rgba(255,255,255,0.03)] p-8 transition-all duration-300 ease-in-out hover:-translate-y-2 hover:shadow-[0_0_40px_rgba(0,255,255,0.2)]">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,255,255,0.8)]" />
                <h3 className="text-2xl md:text-3xl font-semibold tracking-wide">
                  NO₂ [ Nitrogen Dioxide ]
                </h3>
              </div>
              <div className="mt-3 h-0.5 w-12 rounded-full bg-cyan-300/70" />
              <p className="mt-5 text-white/60 leading-relaxed max-w-md">
                Multi-horizon models learn temporal dynamics while graph-aware
                training captures cross-site dependencies for stronger NO₂
                forecasting across the network.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ============ CTA ============ */}
      <section id="cta" className="py-20 px-6 md:px-12">
        <div className="max-w-6xl mx-auto relative">
          <div className="absolute -z-10 right-6 top-1/2 h-72 w-72 -translate-y-1/2 rounded-full bg-cyan-500/10 blur-3xl" />
          <div className="relative rounded-3xl bg-white/5 backdrop-blur-lg border border-white/10 p-10 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_0_30px_rgba(0,255,255,0.15)] overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-transparent rounded-3xl pointer-events-none" />
            <div className="relative z-10">
              <h3 className="text-3xl md:text-4xl font-semibold">
                Explore Forecast Models
              </h3>
              <p className="mt-4 text-white/60 max-w-2xl leading-relaxed">
                Compare residual learning (XGBoost / TCN / LSTM / Transformer) and
                multi-site spatio-temporal forecasting (ST + Graph variants).
                Review metrics and outputs in a clean dashboard flow.
              </p>

              <div className="mt-8 flex flex-wrap gap-4">
                <Link
                  to="/docs"
                  className="rounded-xl bg-white/10 px-6 py-3 text-white/80 ring-1 ring-white/15 hover:bg-white/15 hover:text-white transition"
                >
                  Read Docs
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
