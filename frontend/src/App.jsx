import { useState } from "react";
import "./app.css";


function App() {

  const [text, setText] = useState("");

  const [selectedModel, setSelectedModel] =
    useState("old_muril");

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  const getModelName = (model) => {

    const modelNames = {
      old_muril: "Original MuRIL",
      research_muril: "Research MuRIL",
      indicbert_v2: "IndicBERT v2",
      xlm_roberta: "XLM-RoBERTa",
    };

    return modelNames[model] || model;
  };


  const analyzeSentiment = async () => {

    if (!text.trim()) {

      setError("Please enter a sports comment.");

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


      const data = await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail || "Prediction failed."
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


  const clearAll = () => {

    setText("");

    setResult(null);

    setError("");

  };


  const handleKeyDown = (e) => {

    if (e.ctrlKey && e.key === "Enter") {

      analyzeSentiment();

    }
  };


  return (

    <div className="app-container">

      <main className="sentiment-card">


        <div className="research-badge">

          <span className="research-dot"></span>

          NLP Research Interface

        </div>


        <header className="app-header">

          <h1>
            Hindi Sports Sentiment Analyzer
          </h1>

          <p className="subtitle">

            Analyze Hindi and Hinglish sports comments
            using transformer-based language models.

          </p>

        </header>


        <div className="input-group">

          <label htmlFor="model">

            Select Language Model

          </label>

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

        </div>


        <div className="input-group">

          <label htmlFor="text">

            Sports Comment

          </label>

          <textarea
            id="text"

            rows="6"

            value={text}

            maxLength={500}

            placeholder="Enter a Hindi or Hinglish sports comment..."

            onChange={(e) => {

              setText(
                e.target.value
              );

              if (error) {

                setError("");

              }
            }}

            onKeyDown={handleKeyDown}
          />


          <div className="textarea-meta">

            <span>
              Ctrl + Enter to analyze
            </span>

            <span>
              {text.length}/500
            </span>

          </div>

        </div>


        <div className="button-container">

          <button
            className="analyze-button"

            onClick={analyzeSentiment}

            disabled={
              loading ||
              !text.trim()
            }
          >

            {
              loading
                ? "Analyzing..."
                : "Analyze Sentiment"
            }

          </button>


          <button
            className="clear-button"

            onClick={clearAll}

            disabled={
              loading ||
              (!text && !result)
            }
          >

            Clear

          </button>

        </div>


        {
          error && (

            <div className="error-box">

              <strong>
                Analysis failed
              </strong>

              <div className="error-message">

                {error}

              </div>

            </div>

          )
        }


        {
          loading && (

            <div className="loading-box">

              <div className="loading-title">

                Running sentiment analysis...

              </div>

              <div className="loading-subtitle">

                Using {getModelName(selectedModel)}

              </div>

            </div>

          )
        }


        {
          result && !loading && (

            <section className="result-box">

              <h2>
                Analysis Result
              </h2>


              <div className="result-row">

                <span>
                  Sentiment
                </span>

                <strong>
                  {result.sentiment}
                </strong>

              </div>


              <div className="result-row">

                <span>
                  Confidence
                </span>

                <strong>
                  {result.confidence}%
                </strong>

              </div>


              <div className="result-row">

                <span>
                  Model Used
                </span>

                <strong>
                  {getModelName(result.model)}
                </strong>

              </div>

            </section>

          )
        }


        <footer className="app-footer">

          Hindi & Hinglish Sports Sentiment Analysis

          <span className="footer-dot">
            •
          </span>

          MuRIL · IndicBERT v2 · XLM-RoBERTa

        </footer>


      </main>

    </div>

  );
}


export default App;