import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeSentiment = async () => {
    const cleanedText = text.trim();

    if (!cleanedText) {
      setError("Please enter a Hindi sports sentence.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: cleanedText,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message || "Could not connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setText("");
    setResult(null);
    setError("");
  };

  const getEmoji = (sentiment) => {
    if (sentiment === "Positive") return "🏆";
    if (sentiment === "Negative") return "📉";
    return "⚖️";
  };

  return (
    <main className="app-page">
      <div className="background-glow glow-one"></div>
      <div className="background-glow glow-two"></div>

      <section className="analyzer-card">
        <div className="brand-badge">
          <span>⚡</span>
          MuRIL Powered
        </div>

        <header className="hero">
          <h1>
            Hindi Sports
            <span> Sentiment Analyzer</span>
          </h1>

          <p>
            Analyse Hindi sports comments and instantly classify them
            as positive, negative, or neutral.
          </p>
        </header>

        <div className="input-card">
          <label htmlFor="sportsText">
            Enter Hindi sports text
          </label>

          <textarea
            id="sportsText"
            value={text}
            onChange={(event) =>
              setText(event.target.value)
            }
            placeholder="उदाहरण: भारत ने शानदार जीत हासिल की।"
            maxLength={500}
            disabled={loading}
          />

          <div className="input-footer">
            <span>Hindi and Hinglish supported</span>
            <span>{text.length}/500</span>
          </div>
        </div>

        <div className="action-buttons">
          <button
            className="analyze-button"
            onClick={analyzeSentiment}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analysing sentiment
              </>
            ) : (
              <>
                <span>✨</span>
                Analyse Sentiment
              </>
            )}
          </button>

          <button
            className="clear-button"
            onClick={clearAll}
            disabled={loading || (!text && !result)}
          >
            Clear
          </button>
        </div>

        {error && (
          <div className="error-box">
            <span>⚠️</span>
            {error}
          </div>
        )}

        {result && (
          <div
            className={`result-card ${result.sentiment.toLowerCase()}`}
          >
            <div className="result-heading">
              <div className="sentiment-icon">
                {getEmoji(result.sentiment)}
              </div>

              <div>
                <span className="result-label">
                  Detected sentiment
                </span>
                <h2>{result.sentiment}</h2>
              </div>

              <div className="confidence-score">
                {result.confidence}%
              </div>
            </div>

            <div className="confidence-bar">
              <div
                className="confidence-progress"
                style={{
                  width: `${result.confidence}%`,
                }}
              ></div>
            </div>

            <div className="analysed-text">
              <span>Analysed text</span>
              <p>{result.text}</p>
            </div>
          </div>
        )}

        <footer>
          Fine-tuned using Google MuRIL for Hindi sentiment
          classification
        </footer>
      </section>
    </main>
  );
}

export default App;