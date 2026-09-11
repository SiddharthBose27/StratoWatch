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

const SINGLE_SITE_MODELS = [
  "XGBoost",
  "Random Forest",
  "TCN",
  "Transformer",
  "LSTM",
  "Forecast-only baseline",
];

const MULTI_SITE_MODELS = [
  "ST Transformer",
  "Static Graph-ST",
  "Dynamic Wind Graph-ST",
];

export default function Home() {
  return (
    <div className="bg-black text-white">
      {/* ============ HERO SECTION ============ */}
      <section id="home" className="relative min-h-screen overflow-hidden">
        <div className="absolute inset-0">
          <img
            src={hero}
            className="w-full h-full object-cover"
            alt="StratoWatch atmospheric intelligence"
          />
        </div>

        <div className="absolute inset-0 bg-linear-to-b from-black/50 via-black/70 to-black/95" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_40%,rgba(56,189,248,0.20),transparent_55%)]" />

        <div className="relative z-10 min-h-screen px-6 md:px-12">
          {/* Navbar */}
          <header className="flex items-center justify-between py-8">
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
                A research and forecasting platform for short-term air-quality
                prediction, combining single-site machine learning with
                multi-site spatio-temporal modeling for O₃ and NO₂.
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

              <motion.div
                initial="hidden"
                animate="visible"
                variants={fadeUp}
                custom={4}
                className="mt-10 flex flex-wrap gap-8 text-sm"
              >
                <div>
                  <div className="text-white font-semibold">24h → 6h</div>
                  <div className="text-white/50">Forecast window</div>
                </div>

                <div>
                  <div className="text-white font-semibold">O₃ + NO₂</div>
                  <div className="text-white/50">Target pollutants</div>
                </div>

                <div>
                  <div className="text-white font-semibold">1 + 7 sites</div>
                  <div className="text-white/50">
                    Single & multi-site tracks
                  </div>
                </div>
              </motion.div>

              <div className="mt-14 flex items-center gap-3 text-white/50 text-xs">
                <span className="inline-block h-6 w-px bg-white/20" />
                Scroll to explore
              </div>
            </div>
          </div>
        </div>

        <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-28 bg-linear-to-b from-transparent to-black" />
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
                StratoWatch is an atmospheric intelligence platform built around
                short-term forecasting of O₃ and NO₂. The system evaluates both
                station-level machine-learning models and multi-site
                spatio-temporal architectures using a consistent 24-hour input
                window and 6-hour forecast horizon.
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
                Air-quality forecasting is influenced by temporal patterns,
                meteorological conditions, pollutant persistence, and
                interactions between monitoring locations. StratoWatch studies
                these effects at both individual-station and network scales.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ============ TWO CORE SYSTEMS ============ */}
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
                  Single-Site Forecasting
                </div>

                <p className="mt-3 text-white/60 leading-relaxed">
                  A station-level forecasting track using 24 hours of historical
                  context to predict O₃ and NO₂ over the next 6 hours. Multiple
                  machine-learning and deep-learning models are evaluated
                  against a forecast-only reference baseline.
                </p>

                <div className="mt-6 text-white/70 text-sm font-medium">
                  Models evaluated
                </div>

                <ul className="mt-3 space-y-2 text-white/60 text-sm">
                  {SINGLE_SITE_MODELS.map((model) => (
                    <li key={model}>• {model}</li>
                  ))}
                </ul>

                <div className="mt-6 text-white/45 text-xs">
                  Best overall test performance: XGBoost
                </div>
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
                  A seven-station forecasting system using 24 hours of context
                  to predict the next 6 hours for O₃ and NO₂, with temporal
                  modeling and graph-based spatial variants.
                </p>

                <div className="mt-6 text-white/70 text-sm font-medium">
                  Variants tested
                </div>

                <ul className="mt-3 space-y-2 text-white/60 text-sm">
                  {MULTI_SITE_MODELS.map((model) => (
                    <li key={model}>• {model}</li>
                  ))}
                </ul>

                <div className="mt-6 text-white/45 text-xs">
                  Best MAE: ST Transformer • Best RMSE/R²: Dynamic Wind Graph-ST
                </div>
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
            Final test-set results
          </h2>

          <p className="mt-3 text-white/55 max-w-3xl leading-relaxed">
            These values come from the frozen final evaluation runs. Metrics are
            reported in real target units on held-out test windows.
          </p>

          <div className="mt-10 grid gap-8 lg:grid-cols-2 items-stretch">
            {/* ================== SINGLE SITE ================== */}
            <div className="premium-card p-9 h-full">
              <div className="flex items-start justify-between gap-6">
                <div>
                  <div className="text-white font-semibold text-xl">
                    Single-Site
                  </div>

                  <div className="mt-1 text-white/60 text-sm">
                    Best model: <span className="text-white/80">XGBoost</span>
                  </div>

                  <div className="mt-1 text-white/45 text-xs">
                    24h context • 6h horizon • O₃ + NO₂
                  </div>
                </div>

                <div className="text-[11px] tracking-[0.28em] text-white/45">
                  FROZEN
                </div>
              </div>

              <div className="mt-7 flex justify-center">
                <div className="grid grid-cols-3 gap-4 max-w-2xl w-full">
                  {[
                    { k: "MAE", v: "0.4039" },
                    { k: "RMSE", v: "0.5906" },
                    { k: "R²", v: "0.3848" },
                  ].map((m) => (
                    <div
                      key={m.k}
                      className="metric-tile glow-hover p-5 text-center flex flex-col items-center justify-center"
                    >
                      <div className="text-white/55 text-xs tracking-[0.22em]">
                        {m.k}
                      </div>

                      <div className="mt-3 text-3xl md:text-4xl font-semibold tabular-nums">
                        {m.v}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-8">
                <div className="text-white/60 text-sm">Models evaluated</div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    "XGBoost — best",
                    "Random Forest",
                    "TCN",
                    "Transformer",
                    "LSTM",
                    "Forecast-only",
                  ].map((text) => (
                    <span
                      key={text}
                      className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/60 ring-1 ring-white/10"
                    >
                      {text}
                    </span>
                  ))}
                </div>
              </div>

              <p className="mt-8 text-white/50 text-sm leading-relaxed">
                XGBoost achieved the strongest overall single-site test
                performance among the evaluated models and substantially
                improved over the forecast-only reference.
              </p>
            </div>

            {/* ================== MULTI SITE ================== */}
            <div className="premium-card p-9 h-full">
              <div className="flex items-start justify-between gap-6">
                <div>
                  <div className="text-white font-semibold text-xl">
                    Multi-Site
                  </div>

                  <div className="mt-1 text-white/60 text-sm">
                    Best MAE:{" "}
                    <span className="text-white/80">ST Transformer</span>
                  </div>

                  <div className="mt-1 text-white/45 text-xs">
                    7 sites • 24h context → 6h horizon • O₃ + NO₂
                  </div>
                </div>

                <div className="text-[11px] tracking-[0.28em] text-white/45">
                  FROZEN
                </div>
              </div>

              <div className="mt-7 grid grid-cols-3 gap-4">
                {[
                  { k: "Best MAE", v: "18.8349", model: "ST Transformer" },
                  {
                    k: "Best RMSE",
                    v: "26.9455",
                    model: "Dynamic Graph-ST",
                  },
                  {
                    k: "Best R²",
                    v: "0.3911",
                    model: "Dynamic Graph-ST",
                  },
                ].map((m) => (
                  <div
                    key={m.k}
                    className="metric-tile glow-hover p-5 text-center flex flex-col items-center justify-center"
                  >
                    <div className="text-white/55 text-xs tracking-[0.12em]">
                      {m.k}
                    </div>

                    <div className="mt-3 text-2xl md:text-3xl font-semibold tabular-nums">
                      {m.v}
                    </div>

                    <div className="mt-2 text-white/40 text-[10px] leading-tight">
                      {m.model}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-8">
                <div className="text-white/60 text-sm">Models evaluated</div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    "ST Transformer — MAE 18.8349",
                    "Static Graph-ST — R² 0.3850",
                    "Dynamic Wind Graph-ST — R² 0.3911",
                  ].map((text) => (
                    <span
                      key={text}
                      className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/60 ring-1 ring-white/10"
                    >
                      {text}
                    </span>
                  ))}
                </div>
              </div>

              <p className="mt-8 text-white/50 text-sm leading-relaxed">
                The temporal-only model achieved the lowest MAE, while the
                dynamic wind-conditioned graph model achieved the lowest RMSE
                and highest R². The results show different strengths across
                error measures rather than a single universal winner.
              </p>
            </div>
          </div>

          <p className="mt-8 text-white/40 text-sm">
            All displayed values correspond to the frozen final experiments. See
            Methodology and Docs for evaluation details.
          </p>
        </div>
      </section>

      {/* ============ MONITORING & FORECASTING ============ */}
      <section id="monitoring" className="py-20 px-6 md:px-12">
        <div className="max-w-6xl mx-auto">
          <div className="text-xs tracking-[0.32em] text-cyan-300/80">
            MONITORING & FORECASTING
          </div>

          <h2 className="mt-4 text-3xl md:text-5xl font-semibold tracking-[0.08em]">
            O₃ and NO₂ Monitoring & Forecasting
          </h2>

          <div className="mt-4 h-0.5 w-16 rounded-full bg-cyan-300/80" />

          <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="relative rounded-3xl backdrop-blur-md bg-gradient-to-br from-cyan-500/10 via-transparent to-blue-500/10 border border-white/10 shadow-[inset_0_0_24px_rgba(255,255,255,0.03)] p-8 transition-all duration-300 ease-in-out hover:-translate-y-2 hover:shadow-[0_0_40px_rgba(0,255,255,0.2)]">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,255,255,0.8)]" />

                <h3 className="text-2xl md:text-3xl font-semibold tracking-wide">
                  O₃ [ Ozone ]
                </h3>
              </div>

              <div className="mt-3 h-0.5 w-12 rounded-full bg-cyan-300/70" />

              <p className="mt-5 text-white/60 leading-relaxed max-w-md">
                The system forecasts short-term ozone concentration using
                historical pollutant behavior, meteorological variables, and
                temporal context across the single-site and multi-site tracks.
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
                Multi-horizon forecasting captures temporal dynamics while the
                multi-site experiments examine whether spatial structure helps
                explain variation between monitoring stations.
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
                Explore station-level machine learning alongside multi-site
                spatio-temporal forecasting. Compare model behavior, metrics,
                and evaluation outputs across O₃ and NO₂.
              </p>

              <div className="mt-8 flex flex-wrap gap-4">
                <Link
                  to="/single-site"
                  className="rounded-xl bg-cyan-400 px-6 py-3 text-black font-semibold hover:bg-cyan-300 transition"
                >
                  Single-Site Models
                </Link>

                <Link
                  to="/multi-site"
                  className="rounded-xl bg-white/10 px-6 py-3 text-white/80 ring-1 ring-white/15 hover:bg-white/15 hover:text-white transition"
                >
                  Multi-Site Models
                </Link>

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
