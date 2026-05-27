import { Routes, Route } from "react-router-dom";
import SiteShell from "./components/SiteShell";
import Home from "./pages/Home";
import SingleSite from "./pages/SingleSite";
import MultiSite from "./pages/MultiSite";
import Methodology from "./pages/Methodology";
import Docs from "./pages/Docs";

export default function App() {
  return (
    <SiteShell>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/single-site" element={<SingleSite />} />
        <Route path="/multi-site" element={<MultiSite />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/docs" element={<Docs />} />
      </Routes>
    </SiteShell>
  );
}
