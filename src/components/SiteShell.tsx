import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import footerImg from "../assets/footer.jpg"; // put your footer image in src/assets with this name OR rename import



export default function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Page content */}
      <main className="px-0">{children}</main>

      {/* Footer (global) */}
      <footer className="relative mt-24">
        {/* background image */}
        <div className="absolute inset-0 overflow-hidden opacity-80">
          <img
            src={footerImg}
            alt="Footer background"
            className="h-full w-full object-cover opacity-55"
          />
          {/* overlays for readability */}
          <div className="absolute inset-0 bg-linear-gradient-to-b from-black via-black/80 to-black" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_20%,rgba(56,189,248,0.18),transparent_55%)]" />
        </div>

        <div className="relative mx-auto max-w-6xl px-6 md:px-10 py-16">
          <div className="grid gap-10 md:grid-cols-2">
            <div>
              <div className="flex items-center gap-3">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-white/10 ring-1 ring-white/15">
                  <svg
                    className="h-5 w-5 text-cyan-300 animate-satellite satellite-glow"
                    viewBox="0 0 32 32"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    aria-hidden="true"
                  >
                    <rect x="12" y="12" width="8" height="8" rx="1.6" fill="currentColor" />
                    <rect x="2" y="11" width="7" height="10" rx="1.2" stroke="currentColor" strokeWidth="1.4" />
                    <rect x="23" y="11" width="7" height="10" rx="1.2" stroke="currentColor" strokeWidth="1.4" />
                    <line x1="9" y1="16" x2="12" y2="16" stroke="currentColor" strokeWidth="1.4" />
                    <line x1="20" y1="16" x2="23" y2="16" stroke="currentColor" strokeWidth="1.4" />
                    <circle cx="16" cy="7" r="2" fill="currentColor" />
                    <line x1="16" y1="9.5" x2="16" y2="12" stroke="currentColor" strokeWidth="1.4" />
                    <path d="M22.5 22.5l4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                    <circle cx="27" cy="27" r="2.4" stroke="currentColor" strokeWidth="1.4" />
                  </svg>
                </div>
                <span className="tracking-[0.25em] text-white/80 text-sm">
                  STRATOWATCH
                </span>
              </div>

              <p className="mt-5 max-w-md text-white/60 leading-relaxed text-sm">
                StratoWatch is a forecasting + evaluation hub for air-quality
                intelligence: single-site residual correction and multi-site
                spatio-temporal learning for O₃ and NO₂.
              </p>

              <div className="mt-8">
                <div className="text-white/70 text-sm">
                  Subscribe to updates
                </div>
                <div className="mt-3 flex max-w-md overflow-hidden rounded-xl bg-white/10 ring-1 ring-white/15">
                  <input
                    className="w-full bg-transparent px-4 py-3 text-sm outline-none placeholder:text-white/35"
                    placeholder="Your email"
                  />
                  <button className="px-5 text-sm text-white/80 hover:text-white transition">
                    SUBSCRIBE
                  </button>
                </div>
              </div>
            </div>

            <div className="md:text-right">
              <div className="text-white/60 text-sm">Contact</div>
              <div className="mt-3 flex md:justify-end gap-3 flex-wrap">
                <a
                  href="mailto:bose.siddharth2711@gmail.com"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center rounded-xl bg-white/10 px-3 py-2 text-xs text-white/70 ring-1 ring-white/15 hover:bg-white/15 hover:text-white transition"
                >
                  Email
                </a>
                <a
                  href="https://github.com/SiddharthBose27"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center rounded-xl bg-white/10 px-3 py-2 text-xs text-white/70 ring-1 ring-white/15 hover:bg-white/15 hover:text-white transition"
                >
                  GitHub
                </a>
                <a
                  href="https://www.linkedin.com/in/siddharth-bose-92ab523"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center rounded-xl bg-white/10 px-3 py-2 text-xs text-white/70 ring-1 ring-white/15 hover:bg-white/15 hover:text-white transition"
                >
                  LinkedIn
                </a>
              </div>

              <div className="mt-8 grid gap-2 text-sm text-white/60 md:justify-end">
                <FooterLink to="/" label="Home" />
                <FooterLink to="/single-site" label="Single-Site" />
                <FooterLink to="/multi-site" label="Multi-Site" />
                <FooterLink to="/methodology" label="Methodology" />
                <FooterLink to="/docs" label="Docs" />
              </div>
            </div>
          </div>

          <div className="mt-12 flex flex-wrap items-center justify-between gap-4 border-t border-white/10 pt-8 text-xs text-white/40">
            <div>
              © {new Date().getFullYear()} StratoWatch. All rights reserved.
            </div>
            <div className="flex gap-4">
              <span className="text-white/80 transition">
                @ Site created by Siddharth Bose
              </span>
              <span className="hover:text-white/60 transition">
                Privacy Policy    Terms & Conditions
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}


function FooterLink({ to, label }: { to: string; label: string }) {
  return (
    <NavLink to={to} className="hover:text-white transition">
      {label}
    </NavLink>
  );
}
