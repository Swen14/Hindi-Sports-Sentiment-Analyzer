import { useEffect, useState } from "react";
import {
  Eraser,
  Layers,
  LoaderCircle,
  MessageSquareText,
  Play,
  Target,
} from "lucide-react";

import { GROUP_STYLE, SENTIMENTS, predict } from "../api";
import ModelPicker from "./ModelPicker";


const EXAMPLES = [
  "विराट कोहली ने आज कमाल की पारी खेली, दिल जीत लिया ❤",
  "गंभीर को कोच पद से हटाओ, पूरी टीम बर्बाद कर दी",
  "मैच कल शाम सात बजे शुरू होगा",
  "वाह क्या बैटिंग है, फिर से जीरो पर आउट 😂",
  "भारतीय टीम ने शानदार जीत हासिल की।",
];


export function ProbabilityBars({ probabilities, compact = false }) {
  if (!probabilities) return null;
  return (
    <div className={compact ? "prob-bars compact" : "prob-bars"}>
      {SENTIMENTS.map((label) => (
        <div className="prob-row" key={label}>
          <span className="prob-label">{label}</span>
          <div className="prob-track">
            <div
              className={`prob-fill ${label.toLowerCase()}`}
              style={{ width: `${probabilities[label] ?? 0}%` }}
            />
          </div>
          <span className="prob-value">
            {(probabilities[label] ?? 0).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}


function LiveAnalysis({ models, defaultModel }) {

  const [text, setText] = useState("");
  const [selected, setSelected] = useState(null);
  const [mode, setMode] = useState("single");
  const [result, setResult] = useState(null);
  const [allResults, setAllResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!selected && defaultModel) setSelected(defaultModel);
  }, [defaultModel, selected]);

  const available = models.filter((m) => m.available);
  const selectedModel = models.find((m) => m.id === selected);

  const run = async () => {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    setAllResults([]);

    try {
      if (mode === "single") {
        setResult(await predict(text, selected));
      } else {
        // One at a time: the backend keeps a single model in memory
        for (const model of available) {
          const output = await predict(text, model.id);
          setAllResults((prev) => [...prev, { model, output }]);
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setText("");
    setResult(null);
    setAllResults([]);
    setError("");
  };

  return (
    <section id="analyze" className="section">

      <div className="section-head">
        <span className="eyebrow"><MessageSquareText size={14} /> Live analysis</span>
        <h2>Analyze a sports comment</h2>
        <p>Write a Hindi comment, pick a model, or run all six side by side.</p>
      </div>

      <div className="analyze-grid">

        <div className="card">

          <div className="segmented">
            <button
              className={mode === "single" ? "active" : ""}
              onClick={() => setMode("single")}
            >
              <Target size={15} /> One model
            </button>
            <button
              className={mode === "all" ? "active" : ""}
              onClick={() => setMode("all")}
            >
              <Layers size={15} /> All 6 models
            </button>
          </div>

          {mode === "single" && (
            <ModelPicker
              models={models}
              selected={selected}
              onSelect={(id) => { setSelected(id); setResult(null); }}
              compact
            />
          )}

          <label className="field-label" htmlFor="comment">Sports comment</label>
          <textarea
            id="comment"
            value={text}
            maxLength={500}
            placeholder="जैसे: रोहित शर्मा ने शानदार कप्तानी की..."
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.ctrlKey && e.key === "Enter") run();
            }}
          />
          <div className="textarea-meta">
            <span>Ctrl + Enter to analyze</span>
            <span>{text.length}/500</span>
          </div>

          <div className="examples">
            {EXAMPLES.map((example) => (
              <button key={example} onClick={() => setText(example)}>
                {example}
              </button>
            ))}
          </div>

          <div className="actions">
            <button
              className="btn btn-primary"
              onClick={run}
              disabled={loading || !text.trim() || (mode === "single" && !selected)}
            >
              {loading
                ? <><LoaderCircle size={17} className="spin" /> Analyzing</>
                : <><Play size={16} fill="currentColor" /> Analyze</>}
            </button>
            <button
              className="btn btn-ghost"
              onClick={clear}
              disabled={loading || (!text && !result && !allResults.length)}
            >
              <Eraser size={16} /> Clear
            </button>
          </div>

          {error && <div className="banner banner-error">{error}</div>}
        </div>


        <div className="card result-card">

          {mode === "single" && !result && !loading && (
            <div className="empty">
              <MessageSquareText size={30} />
              <strong>Prediction appears here</strong>
              <span>
                {selectedModel
                  ? `Using ${selectedModel.name} · ${GROUP_STYLE[selectedModel.group].label}`
                  : "Select a model to begin"}
              </span>
            </div>
          )}

          {mode === "single" && loading && (
            <div className="empty">
              <LoaderCircle size={30} className="spin" />
              <strong>Running {selectedModel?.name}</strong>
              <span>The first request loads the model into memory</span>
            </div>
          )}

          {mode === "single" && result && !loading && (
            <div className="result">
              <div className={`verdict ${result.sentiment.toLowerCase()}`}>
                <span>Detected sentiment</span>
                <strong>{result.sentiment}</strong>
                <em>{result.confidence}% confidence</em>
              </div>
              <ProbabilityBars probabilities={result.probabilities} />
              <div className="result-meta">
                <span className={`tag tag-${GROUP_STYLE[selectedModel?.group]?.tone}`}>
                  {GROUP_STYLE[selectedModel?.group]?.label}
                </span>
                <span>{selectedModel?.name}</span>
              </div>
            </div>
          )}

          {mode === "all" && !allResults.length && !loading && (
            <div className="empty">
              <Layers size={30} />
              <strong>Run every model on the same comment</strong>
              <span>See where synthetic-data and real-data models disagree</span>
            </div>
          )}

          {mode === "all" && (allResults.length > 0 || loading) && (
            <div className="all-results">
              {available.map((model) => {
                const row = allResults.find((r) => r.model.id === model.id);
                return (
                  <div className="all-row" key={model.id}>
                    <div className="all-row-head">
                      <span className={`dot dot-${GROUP_STYLE[model.group].tone}`} />
                      <strong>{model.name}</strong>
                      <small>{GROUP_STYLE[model.group].short}</small>
                      {row
                        ? <span className={`pill ${row.output.sentiment.toLowerCase()}`}>
                            {row.output.sentiment} · {row.output.confidence}%
                          </span>
                        : <LoaderCircle size={15} className={loading ? "spin muted" : "muted"} />}
                    </div>
                    {row && <ProbabilityBars probabilities={row.output.probabilities} compact />}
                  </div>
                );
              })}
            </div>
          )}
        </div>

      </div>
    </section>
  );
}


export default LiveAnalysis;
