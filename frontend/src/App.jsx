import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:5000/predict";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const examples = [
    "भारतीय टीम ने शानदार जीत हासिल की।",
    "टीम का प्रदर्शन बहुत निराशाजनक था।",
    "मैच कल शाम सात बजे शुरू होगा।",
  ];

  const handleAnalyze = async () => {
    const cleanedText = text.trim();

    if (!cleanedText) {
      setError("Please enter some Hindi sports text.");
      setResult(null);
      return;
    }

    if (cleanedText.length < 3) {
      setError("Please enter a longer sentence.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: cleanedText,
        }),
      });

      if (!response.ok) {
        let message = "The server returned an invalid response.";

        try {
          const errorData = await response.json();

          message =
            errorData.detail ||
            errorData.message ||
            errorData.error ||
            message;
        } catch {
          // Use the default message if the response is not JSON.
        }

        throw new Error(message);
      }

      const data = await response.json();

      if (!data.sentiment) {
        throw new Error("The prediction response does not contain sentiment.");
      }

      setResult({
        sentiment: data.sentiment,
        confidence: data.confidence,
      });
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err.message ||
          "Prediction failed. Make sure the backend server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText("");
    setResult(null);
    setError("");
  };

  const handleExample = (example) => {
    setText(example);
    setResult(null);
    setError("");
  };

  const getSentimentDetails = (sentiment) => {
    const normalizedSentiment = String(sentiment || "")
      .trim()
      .toLowerCase();

    if (
      normalizedSentiment === "positive" ||
      normalizedSentiment === "सकारात्मक"
    ) {
      return {
        label: "Positive",
        icon: "😊",
        className: "positive",
        message: "The entered text expresses a positive sentiment.",
      };
    }

    if (
      normalizedSentiment === "negative" ||
      normalizedSentiment === "नकारात्मक"
    ) {
      return {
        label: "Negative",
        icon: "😞",
        className: "negative",
        message: "The entered text expresses a negative sentiment.",
      };
    }

    return {
      label: "Neutral",
      icon: "😐",
      className: "neutral",
      message: "The entered text expresses a neutral sentiment.",
    };
  };

  const sentimentDetails = result
    ? getSentimentDetails(result.sentiment)
    : null;

  const getConfidencePercentage = (confidence) => {
    const value = Number(confidence);

    if (!Number.isFinite(value)) {
      return null;
    }

    const percentage = value <= 1 ? value * 100 : value;

    return Math.min(Math.max(percentage, 0), 100);
  };

  const confidencePercentage = result
    ? getConfidencePercentage(result.confidence)
    : null;

  return (
    <div className="app-page">
      <div className="background-glow glow-one"></div>
      <div className="background-glow glow-two"></div>

      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">HS</div>

          <div className="brand-text">
            <h1>Hindi Sports Sentiment Analyzer</h1>
            <p>Sports text sentiment classification</p>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          System Ready
        </div>
      </header>

      <main className="main-content">
        <section className="hero-section">
          <div className="hero-badge">Hindi Sports Sentiment Analysis</div>

          <h2>Understand the sentiment behind Hindi sports text</h2>

          <p>
            Enter a Hindi sports-related sentence, news headline, comment or
            reaction to classify it as positive, negative or neutral.
          </p>
        </section>

        <section className="analyzer-card">
          <div className="input-header">
            <div>
              <h3>Enter Hindi Sports Text</h3>

              <p>
                Write a Hindi sports headline, statement, comment or reaction.
              </p>
            </div>

            <span className="character-count">{text.length}/500</span>
          </div>

          <textarea
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setError("");
            }}
            placeholder="Example: भारतीय टीम ने शानदार प्रदर्शन किया।"
            maxLength={500}
            rows={7}
            disabled={loading}
          />

          <div className="example-section">
            <p className="example-title">Try an example:</p>

            <div className="example-list">
              {examples.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="example-button"
                  onClick={() => handleExample(example)}
                  disabled={loading}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {error && (
            <div className="error-message" role="alert">
              {error}
            </div>
          )}

          <div className="button-row">
            <button
              type="button"
              className="clear-button"
              onClick={handleClear}
              disabled={loading || (!text && !result && !error)}
            >
              Clear
            </button>

            <button
              type="button"
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                "Analyze Sentiment"
              )}
            </button>
          </div>
        </section>

        {result && sentimentDetails && (
          <section className={`result-card ${sentimentDetails.className}`}>
            <div className="result-icon">{sentimentDetails.icon}</div>

            <div className="result-content">
              <p className="result-heading">Detected Sentiment</p>

              <h3>{sentimentDetails.label}</h3>

              <p className="result-message">
                {sentimentDetails.message}
              </p>

              {confidencePercentage !== null && (
                <div className="confidence-section">
                  <div className="confidence-header">
                    <span>Confidence Score</span>

                    <strong>{confidencePercentage.toFixed(2)}%</strong>
                  </div>

                  <div className="confidence-track">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${confidencePercentage}%`,
                      }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        <section className="information-grid">
          <article className="info-card">
            <div className="info-icon">😊</div>

            <h3>Positive</h3>

            <p>
              Represents victory, praise, excitement, confidence and strong
              performance.
            </p>
          </article>

          <article className="info-card">
            <div className="info-icon">😐</div>

            <h3>Neutral</h3>

            <p>
              Represents factual sports information without a strong positive
              or negative opinion.
            </p>
          </article>

          <article className="info-card">
            <div className="info-icon">😞</div>

            <h3>Negative</h3>

            <p>
              Represents defeat, disappointment, criticism and poor
              performance.
            </p>
          </article>
        </section>
      </main>

      <footer className="footer">
        <p>Hindi Sports Sentiment Analyzer</p>
        <span>Academic Natural Language Processing Project</span>
      </footer>
    </div>
  );
}

export default App;