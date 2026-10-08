import { useEffect, useState } from "react";
import {
  BarChart3,
  BrainCircuit,
  Database,
  FlaskConical,
  GitBranch,
  MessageSquareText,
  Sparkles,
} from "lucide-react";

import "./App.css";
import { fetchModels } from "./api";
import LiveAnalysis from "./components/LiveAnalysis";
import ModelLab from "./components/ModelLab";
import Comparison from "./components/Comparison";


const NAV = [
  { id: "analyze", label: "Analyze", icon: MessageSquareText },
  { id: "lab", label: "Model Lab", icon: FlaskConical },
  { id: "compare", label: "Compare", icon: BarChart3 },
];


function App() {

  const [catalog, setCatalog] = useState(null);
  const [catalogError, setCatalogError] = useState("");
  const [active, setActive] = useState("");

  useEffect(() => {
    fetchModels()
      .then(setCatalog)
      .catch((err) => setCatalogError(err.message));
  }, []);

  // Highlight the nav item for the section in view
  useEffect(() => {
    const onScroll = () => {
      const line = window.innerHeight * 0.4;
      const current = NAV.filter(({ id }) => {
        const el = document.getElementById(id);
        return el && el.getBoundingClientRect().top < line;
      }).pop();
      setActive(current ? current.id : "");
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [catalog]);

  const models = catalog?.models || [];

  return (
    <div className="app">

      <header className="topnav">
        <div className="topnav-inner">
          <a className="brand" href="#top">
            <span className="brand-mark"><BrainCircuit size={20} /></span>
            <span>
              <strong>Hindi Sports Sentiment</strong>
              <small>Analyzer · NLP research</small>
            </span>
          </a>

          <nav className="nav-links">
            {NAV.map(({ id, label, icon: Icon }) => (
              <a
                key={id}
                href={`#${id}`}
                className={active === id ? "active" : ""}
              >
                <Icon size={16} />
                <span>{label}</span>
              </a>
            ))}
          </nav>

          <a
            className="nav-github"
            href="https://github.com/Swen14/Hindi-Sports-Sentiment-Analyzer"
            target="_blank"
            rel="noreferrer"
            title="GitHub repository"
          >
            <GitBranch size={18} />
          </a>
        </div>
      </header>


      <main id="top">

        <section className="hero">
          <div className="hero-badge">
            <Sparkles size={14} />
            3 architectures × 2 training datasets
          </div>

          <h1>
            How well does sentiment AI read
            <span className="gradient-text"> real Hindi sports fans?</span>
          </h1>

          <p>
            MuRIL, IndicBERT v2 and XLM-RoBERTa, each fine-tuned twice:
            once on AI-generated sentences and once on real YouTube comments.
            Try them live, inspect every parameter, and compare how they
            hold up on real-world text.
          </p>

          <div className="hero-stats">
            <div>
              <Database size={18} />
              <strong>10,000</strong>
              <span>synthetic sentences</span>
            </div>
            <div>
              <MessageSquareText size={18} />
              <strong>3,720</strong>
              <span>real comments, verified</span>
            </div>
            <div>
              <BrainCircuit size={18} />
              <strong>6</strong>
              <span>fine-tuned models</span>
            </div>
          </div>

          {catalogError && (
            <div className="banner banner-error">{catalogError}</div>
          )}
        </section>


        <LiveAnalysis
          models={models}
          defaultModel={catalog?.default_model}
        />

        <ModelLab
          models={models}
          groups={catalog?.groups}
          testSets={catalog?.test_sets}
        />

        <Comparison models={models} />

      </main>


      <footer className="footer">
        <span>Hindi Sports Sentiment Analyzer</span>
        <span>MuRIL · IndicBERT v2 · XLM-RoBERTa</span>
        <span>Swen Lemos · St. Francis Institute of Technology</span>
      </footer>

    </div>
  );
}


export default App;
