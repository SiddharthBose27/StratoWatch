import { NavLink, useNavigate } from "react-router-dom";

function cx(...classes: Array<string | false | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cx(
          "transition nav-link",
          isActive ? "text-white" : "text-white/70 hover:text-white",
        )
      }
    >
      {label}
    </NavLink>
  );
}

export default function Navbar() {
  const navigate = useNavigate();
  return (
    <div className="fixed top-0 left-0 right-0 z-50 px-6 md:px-10 pt-6">
      <header className="mx-auto max-w-6xl rounded-2xl bg-white/5 ring-1 ring-white/10 backdrop-blur-xl">
        <div className="flex items-center justify-between px-5 py-4">
          {/* Brand */}
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

          {/* Links */}
          <nav className="hidden md:flex items-center gap-8 text-sm">
            <NavItem to="/" label="Home" />
            <NavItem to="/single-site" label="Single-Site" />
            <NavItem to="/multi-site" label="Multi-Site" />
            <NavItem to="/methodology" label="Methodology" />
            <NavItem to="/docs" label="Docs" />
          </nav>

          {/* CTA */}
          <button
            onClick={() => navigate("/docs#contact")}
            className="rounded-xl bg-white/10 px-4 py-2 text-sm text-white/80 ring-1 ring-white/15 hover:bg-white/15 hover:text-white transition"
          >
            Get in touch
          </button>
        </div>
      </header>
    </div>
  );
}
