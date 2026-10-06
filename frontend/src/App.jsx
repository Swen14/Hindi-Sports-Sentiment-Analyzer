import { useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Cpu,
  Eraser,
  Gauge,
  LayoutDashboard,
  MessageSquareText,
  Microscope,
  Network,
  Play,
  Sparkles,
  Target,
  Zap,
} from "lucide-react";

import "./App.css";
import ModelEvaluation from "./ModelEvaluation";


function App() {

  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState("old_muril");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [activeSection, setActiveSection] =
    useState("analysis");


  const modelNames = {

    old_muril: "Original MuRIL",

    research_muril: "Research MuRIL",

    indicbert_v2: "IndicBERT v2",

    xlm_roberta: "XLM-RoBERTa",

  };


  const getModelName = (model) => {

    return modelNames[model] || model;

  };


  const scrollToSection = (section) => {

    setActiveSection(section);

    const element =
      document.getElementById(section);

    if (element) {

      element.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });

    }

  };


  // ============================================================
  // SENTIMENT ANALYSIS
  // ============================================================

  const analyzeSentiment = async () => {

    if (!text.trim()) {

      setError(
        "Please enter a sports comment."
      );

      return;

    }


    setLoading(true);
    setError("");
    setResult(null);


    try {

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            text: text,
            model: selectedModel,
          }),

        }
      );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Prediction failed."
        );

      }


      setResult(data);

    } catch (err) {

      console.error(err);

      setError(
        err.message ||
        "Could not connect to the backend."
      );

    } finally {

      setLoading(false);

    }

  };


  // ============================================================
  // CLEAR
  // ============================================================

  const clearAll = () => {

    setText("");
    setResult(null);
    setError("");

  };


  // ============================================================
  // CTRL + ENTER
  // ============================================================

  const handleKeyDown = (e) => {

    if (
      e.ctrlKey &&
      e.key === "Enter"
    ) {

      analyzeSentiment();

    }

  };


  // ============================================================
  // RESULT COLOR
  // ============================================================

  const getSentimentClass = (sentiment) => {

    if (!sentiment) {
      return "";
    }

    const value =
      sentiment.toLowerCase();

    if (
      value.includes("positive") ||
      value.includes("सकारात्मक")
    ) {

      return "positive";

    }

    if (
      value.includes("negative") ||
      value.includes("नकारात्मक")
    ) {

      return "negative";

    }

    return "neutral";

  };


  return (

    <div className="dashboard-shell">

      {/* ========================================================
          BACKGROUND EFFECTS
          ======================================================== */}

      <div className="background-grid"></div>

      <div className="ambient-glow glow-one"></div>

      <div className="ambient-glow glow-two"></div>


      {/* ========================================================
          SIDEBAR
          ======================================================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">

            <BrainCircuit size={25} />

          </div>

          <div>

            <div className="brand-name">
              SPORT<span>AI</span>
            </div>

            <div className="brand-subtitle">
              NLP RESEARCH
            </div>

          </div>

        </div>


        <div className="sidebar-line"></div>


        <nav className="sidebar-nav">

          <button
            className={
              activeSection === "overview"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() =>
              scrollToSection("overview")
            }
          >

            <LayoutDashboard size={18} />

            <span>
              Overview
            </span>

          </button>


          <button
            className={
              activeSection === "analysis"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() =>
              scrollToSection("analysis")
            }
          >

            <MessageSquareText size={18} />

            <span>
              Sentiment Analysis
            </span>

          </button>


          <button
            className={
              activeSection === "performance"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() =>
              scrollToSection("performance")
            }
          >

            <BarChart3 size={18} />

            <span>
              Model Performance
            </span>

          </button>


          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("models")
            }
          >

            <Cpu size={18} />

            <span>
              Research Models
            </span>

          </button>

        </nav>


        <div className="sidebar-bottom">

          <div className="system-status">

            <span className="status-pulse"></span>

            <div>

              <strong>
                System Online
              </strong>

              <small>
                Backend connected
              </small>

            </div>

          </div>


          <div className="sidebar-version">

            v3.0 · Hindi Sports NLP

          </div>

        </div>

      </aside>


      {/* ========================================================
          MAIN CONTENT
          ======================================================== */}

      <main className="main-content">


        {/* ======================================================
            TOP HEADER
            ====================================================== */}

        <header className="topbar">

          <div>

            <div className="topbar-eyebrow">

              <CircleDot size={13} />

              AI SENTIMENT INTELLIGENCE

            </div>

            <h1>
              Hindi Sports
              <span> Sentiment Analyzer</span>
            </h1>

          </div>


          <div className="topbar-right">

            <div className="live-indicator">

              <span></span>

              LIVE

            </div>


            <div className="topbar-model">

              <Cpu size={15} />

              {getModelName(selectedModel)}

            </div>

          </div>

        </header>


        {/* ======================================================
            OVERVIEW
            ====================================================== */}

        <section
          id="overview"
          className="dashboard-section overview-section"
        >

          <div className="section-heading">

            <div>

              <div className="section-kicker">
                <Sparkles size={14} />
                OVERVIEW
              </div>

              <h2>
                Sentiment intelligence for
                <span> sports conversations.</span>
              </h2>

              <p>
                Analyze Hindi and Hinglish sports
                comments using transformer-based
                language models.
              </p>

            </div>

          </div>


          <div className="overview-grid">

            <div className="overview-card">

              <div className="overview-card-icon">
                <MessageSquareText size={21} />
              </div>

              <div>

                <span>
                  INPUT LANGUAGE
                </span>

                <strong>
                  Hindi + Hinglish
                </strong>

              </div>

            </div>


            <div className="overview-card">

              <div className="overview-card-icon">
                <Network size={21} />
              </div>

              <div>

                <span>
                  ARCHITECTURES
                </span>

                <strong>
                  4 Transformer Models
                </strong>

              </div>

            </div>


            <div className="overview-card">

              <div className="overview-card-icon">
                <Target size={21} />
              </div>

              <div>

                <span>
                  TASK
                </span>

                <strong>
                  3-Class Sentiment
                </strong>

              </div>

            </div>


            <div className="overview-card">

              <div className="overview-card-icon">
                <Activity size={21} />
              </div>

              <div>

                <span>
                  OUTPUT
                </span>

                <strong>
                  Sentiment + Confidence
                </strong>

              </div>

            </div>

          </div>

        </section>


        {/* ======================================================
            SENTIMENT ANALYSIS
            ====================================================== */}

        <section
          id="analysis"
          className="dashboard-section"
        >

          <div className="section-heading">

            <div>

              <div className="section-kicker">
                <Zap size={14} />
                LIVE ANALYSIS
              </div>

              <h2>
                Analyze a sports comment
              </h2>

              <p>
                Select a language model and enter
                a Hindi or Hinglish sports comment.
              </p>

            </div>

            <div className="section-badge">

              <span></span>

              READY

            </div>

          </div>


          <div className="analysis-layout">


            {/* INPUT PANEL */}

            <div className="glass-panel analysis-panel">

              <div className="panel-header">

                <div>

                  <span className="panel-label">
                    INPUT CONFIGURATION
                  </span>

                  <h3>
                    Sentiment Analysis
                  </h3>

                </div>

                <div className="panel-icon">
                  <MessageSquareText size={20} />
                </div>

              </div>


              {/* MODEL */}

              <div className="field">

                <label htmlFor="model">
                  LANGUAGE MODEL
                </label>

                <div className="select-wrapper">

                  <Cpu size={17} />

                  <select
                    id="model"
                    value={selectedModel}
                    onChange={(e) => {

                      setSelectedModel(
                        e.target.value
                      );

                      setResult(null);
                      setError("");

                    }}
                  >

                    <option value="old_muril">
                      Original MuRIL
                    </option>

                    <option value="research_muril">
                      Research MuRIL
                    </option>

                    <option value="indicbert_v2">
                      IndicBERT v2
                    </option>

                    <option value="xlm_roberta">
                      XLM-RoBERTa
                    </option>

                  </select>

                  <ChevronDown
                    size={16}
                    className="select-arrow"
                  />

                </div>

              </div>


              {/* TEXT */}

              <div className="field">

                <div className="field-label-row">

                  <label htmlFor="text">
                    SPORTS COMMENT
                  </label>

                  <span>
                    Hindi / Hinglish
                  </span>

                </div>


                <textarea
                  id="text"
                  value={text}
                  maxLength={500}
                  placeholder="Enter a Hindi or Hinglish sports comment..."
                  onChange={(e) => {

                    setText(e.target.value);

                    if (error) {
                      setError("");
                    }

                  }}
                  onKeyDown={handleKeyDown}
                />

                <div className="textarea-footer">

                  <span>
                    CTRL + ENTER TO ANALYZE
                  </span>

                  <span>
                    {text.length}/500
                  </span>

                </div>

              </div>


              {/* BUTTONS */}

              <div className="action-row">

                <button
                  className="primary-button"
                  onClick={analyzeSentiment}
                  disabled={
                    loading ||
                    !text.trim()
                  }
                >

                  {loading ? (

                    <>
                      <span className="button-spinner"></span>
                      ANALYZING...
                    </>

                  ) : (

                    <>
                      <Play size={17} fill="currentColor" />
                      ANALYZE SENTIMENT
                    </>

                  )}

                </button>


                <button
                  className="secondary-button"
                  onClick={clearAll}
                  disabled={
                    loading ||
                    (!text && !result)
                  }
                >

                  <Eraser size={17} />

                  CLEAR

                </button>

              </div>


              {/* ERROR */}

              {error && (

                <div className="error-box">

                  <strong>
                    ANALYSIS FAILED
                  </strong>

                  <span>
                    {error}
                  </span>

                </div>

              )}

            </div>


            {/* RESULT PANEL */}

            <div className="glass-panel result-panel">

              <div className="panel-header">

                <div>

                  <span className="panel-label">
                    MODEL OUTPUT
                  </span>

                  <h3>
                    Analysis Result
                  </h3>

                </div>

                <div className="panel-icon">
                  <Gauge size={20} />
                </div>

              </div>


              {loading && (

                <div className="empty-state">

                  <div className="analysis-orbit">

                    <div></div>

                  </div>

                  <strong>
                    Processing comment
                  </strong>

                  <span>
                    Running {getModelName(selectedModel)}
                  </span>

                </div>

              )}


              {!loading && !result && (

                <div className="empty-state">

                  <div className="empty-icon">

                    <BrainCircuit size={31} />

                  </div>

                  <strong>
                    Awaiting input
                  </strong>

                  <span>
                    Your sentiment prediction
                    will appear here.
                  </span>

                </div>

              )}


              {result && !loading && (

                <div className="result-content">

                  <div
                    className={
                      `sentiment-display ${
                        getSentimentClass(
                          result.sentiment
                        )
                      }`
                    }
                  >

                    <div className="sentiment-icon">

                      <CheckCircle2 size={25} />

                    </div>

                    <div>

                      <span>
                        DETECTED SENTIMENT
                      </span>

                      <strong>
                        {result.sentiment}
                      </strong>

                    </div>

                  </div>


                  <div className="confidence-section">

                    <div className="confidence-header">

                      <span>
                        CONFIDENCE
                      </span>

                      <strong>
                        {result.confidence}%
                      </strong>

                    </div>

                    <div className="confidence-track">

                      <div
                        className="confidence-fill"
                        style={{
                          width:
                            `${result.confidence}%`,
                        }}
                      ></div>

                    </div>

                  </div>


                  <div className="result-details">

                    <div>

                      <span>
                        MODEL
                      </span>

                      <strong>
                        {getModelName(
                          result.model
                        )}
                      </strong>

                    </div>

                    <div>

                      <span>
                        INPUT
                      </span>

                      <strong>
                        {text.length} characters
                      </strong>

                    </div>

                  </div>

                </div>

              )}

            </div>

          </div>

        </section>


        {/* ======================================================
            MODEL PERFORMANCE
            ====================================================== */}

        <section
          id="performance"
          className="dashboard-section performance-section"
        >

          <ModelEvaluation />

        </section>


        {/* ======================================================
            RESEARCH MODELS
            ====================================================== */}

        <section
          id="models"
          className="dashboard-section"
        >

          <div className="section-heading">

            <div>

              <div className="section-kicker">

                <Microscope size={14} />

                RESEARCH MODELS

              </div>

              <h2>
                Transformer model suite
              </h2>

              <p>
                Compare the models integrated
                into the sentiment analysis pipeline.
              </p>

            </div>

          </div>


          <div className="model-grid">

            <div className="model-card">

              <div className="model-number">
                01
              </div>

              <div className="model-card-icon">
                <BrainCircuit size={21} />
              </div>

              <h3>
                Original MuRIL
              </h3>

              <p>
                Baseline multilingual representation
                model optimized for Indian languages.
              </p>

              <span className="model-tag">
                BASELINE
              </span>

            </div>


            <div className="model-card featured">

              <div className="model-number">
                02
              </div>

              <div className="model-card-icon">
                <Sparkles size={21} />
              </div>

              <h3>
                Research MuRIL
              </h3>

              <p>
                Fine-tuned MuRIL model trained for
                the Hindi sports sentiment task.
              </p>

              <span className="model-tag">
                FINE-TUNED
              </span>

            </div>


            <div className="model-card">

              <div className="model-number">
                03
              </div>

              <div className="model-card-icon">
                <Network size={21} />
              </div>

              <h3>
                IndicBERT v2
              </h3>

              <p>
                Indic-language transformer evaluated
                on the same sentiment benchmark.
              </p>

              <span className="model-tag">
                RESEARCH
              </span>

            </div>


            <div className="model-card">

              <div className="model-number">
                04
              </div>

              <div className="model-card-icon">
                <Cpu size={21} />
              </div>

              <h3>
                XLM-RoBERTa
              </h3>

              <p>
                Multilingual transformer evaluated
                for cross-lingual sports sentiment.
              </p>

              <span className="model-tag">
                RESEARCH
              </span>

            </div>

          </div>

        </section>


        {/* ======================================================
            FOOTER
            ====================================================== */}

        <footer className="dashboard-footer">

          <div>

            <BrainCircuit size={17} />

            <span>
              HINDI SPORTS SENTIMENT ANALYZER
            </span>

          </div>

          <span>
            MuRIL · IndicBERT v2 · XLM-RoBERTa
          </span>

          <span>
            NLP RESEARCH INTERFACE · v3.0
          </span>

        </footer>

      </main>

    </div>

  );

}


export default App;