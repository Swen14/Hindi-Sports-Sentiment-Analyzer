import { useEffect, useState } from "react";


function ModelEvaluation() {

  const [selectedModel, setSelectedModel] =
    useState("old_muril");

  const [evaluation, setEvaluation] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");


  const modelNames = {

    old_muril:
      "Original MuRIL",

    research_muril:
      "Research MuRIL",

    indicbert_v2:
      "IndicBERT v2",

    xlm_roberta:
      "XLM-RoBERTa"

  };


  // ============================================================
  // LOAD EVALUATION
  // ============================================================

  const loadEvaluation = async (model) => {

    setLoading(true);
    setError("");
    setEvaluation(null);


    try {

      const response = await fetch(
        `http://127.0.0.1:8000/evaluation/${model}`
      );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Could not load evaluation."
        );

      }


      setEvaluation(data);


    } catch (err) {

      console.error(err);

      setError(
        err.message ||
        "Could not connect to backend."
      );

    } finally {

      setLoading(false);

    }

  };


  // ============================================================
  // INITIAL LOAD
  // ============================================================

  useEffect(() => {

    loadEvaluation(
      selectedModel
    );

  }, []);


  // ============================================================
  // MODEL CHANGE
  // ============================================================

  const handleModelChange = (e) => {

    const model =
      e.target.value;

    setSelectedModel(model);

    loadEvaluation(model);

  };


  // ============================================================
  // FORMAT PERCENT
  // ============================================================

  const formatPercent = (value) => {

    if (
      value === null ||
      value === undefined
    ) {

      return "N/A";

    }

    return (
      Number(value) * 100
    ).toFixed(2) + "%";

  };


  // ============================================================
  // ROC SVG
  // ============================================================

  const drawRoc = (rocData) => {

    if (!rocData) {
      return null;
    }


    const width = 600;
    const height = 380;

    const left = 70;
    const right = 30;
    const top = 25;
    const bottom = 60;

    const graphWidth =
      width - left - right;

    const graphHeight =
      height - top - bottom;


    const convertPoints = (points) => {

      if (!points || points.length === 0) {
        return "";
      }


      return points
        .map((point) => {

          const x =
            left +
            point.fpr *
            graphWidth;

          const y =
            top +
            (1 - point.tpr) *
            graphHeight;

          return `${x},${y}`;

        })
        .join(" ");

    };


    return (

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="roc-svg"
      >

        {/* Grid */}

        <line
          x1={left}
          y1={top}
          x2={left}
          y2={height - bottom}
          className="roc-axis"
        />

        <line
          x1={left}
          y1={height - bottom}
          x2={width - right}
          y2={height - bottom}
          className="roc-axis"
        />


        {/* Random classifier */}

        <line
          x1={left}
          y1={height - bottom}
          x2={width - right}
          y2={top}
          className="roc-random"
        />


        {/* Negative */}

        {rocData.negative && (

          <polyline
            points={convertPoints(
              rocData.negative
            )}
            className="roc-negative"
          />

        )}


        {/* Neutral */}

        {rocData.neutral && (

          <polyline
            points={convertPoints(
              rocData.neutral
            )}
            className="roc-neutral"
          />

        )}


        {/* Positive */}

        {rocData.positive && (

          <polyline
            points={convertPoints(
              rocData.positive
            )}
            className="roc-positive"
          />

        )}


        {/* Axis labels */}

        <text
          x={width / 2}
          y={height - 15}
          textAnchor="middle"
          className="roc-label"
        >
          False Positive Rate
        </text>


        <text
          x="18"
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 18 ${
            height / 2
          })`}
          className="roc-label"
        >
          True Positive Rate
        </text>


        {/* Tick labels */}

        <text
          x={left - 5}
          y={height - bottom + 20}
          className="roc-tick"
        >
          0
        </text>

        <text
          x={width - right - 5}
          y={height - bottom + 20}
          className="roc-tick"
        >
          1
        </text>

        <text
          x={left - 20}
          y={top + 5}
          className="roc-tick"
        >
          1
        </text>

        <text
          x={left - 20}
          y={height - bottom + 5}
          className="roc-tick"
        >
          0
        </text>

      </svg>

    );

  };


  // ============================================================
  // UI
  // ============================================================

  return (

    <section className="evaluation-section">


      <div className="evaluation-header">

        <span className="evaluation-label">
          MODEL EVALUATION
        </span>

        <h2>
          Model Performance
        </h2>

        <p>
          Precision, recall, F1 score, accuracy,
          confusion matrix and ROC-AUC evaluation.
        </p>

      </div>


      {/* MODEL SELECTOR */}

      <div className="evaluation-selector">

        <label htmlFor="evaluation-model">

          Select Model

        </label>


        <select
          id="evaluation-model"
          value={selectedModel}
          onChange={handleModelChange}
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


      {/* LOADING */}

      {loading && (

        <div className="evaluation-loading">

          Evaluating{" "}

          <strong>
            {modelNames[selectedModel]}
          </strong>

          ...

        </div>

      )}


      {/* ERROR */}

      {error && !loading && (

        <div className="evaluation-error">

          <strong>
            Evaluation failed
          </strong>

          <p>
            {error}
          </p>

        </div>

      )}


      {/* RESULTS */}

      {evaluation && !loading && (

        <div className="evaluation-results">


          <div className="evaluation-model-name">

            <span>
              Currently Selected Model
            </span>

            <strong>
              {modelNames[selectedModel]}
            </strong>

          </div>


          {/* ==================================================
              MAIN METRICS
              ================================================== */}

          <div className="metric-grid">

            <div className="metric-card">

              <span>
                Accuracy
              </span>

              <strong>
                {formatPercent(
                  evaluation.accuracy
                )}
              </strong>

            </div>


            <div className="metric-card">

              <span>
                Precision
              </span>

              <strong>
                {formatPercent(
                  evaluation.macro_precision
                )}
              </strong>

            </div>


            <div className="metric-card">

              <span>
                Recall
              </span>

              <strong>
                {formatPercent(
                  evaluation.macro_recall
                )}
              </strong>

            </div>


            <div className="metric-card">

              <span>
                F1 Score
              </span>

              <strong>
                {formatPercent(
                  evaluation.macro_f1
                )}
              </strong>

            </div>

          </div>


          {/* ==================================================
              DATASET INFO
              ================================================== */}

          <div className="evaluation-info">

            <div>

              <span>
                Test Samples
              </span>

              <strong>
                {evaluation.test_samples}
              </strong>

            </div>


            <div>

              <span>
                Dataset
              </span>

              <strong>
                {evaluation.dataset}
              </strong>

            </div>

          </div>


          {/* ==================================================
              CLASSIFICATION REPORT
              ================================================== */}

          {evaluation.classification_report && (

            <div className="evaluation-panel">

              <h3>
                Classification Report
              </h3>


              <div className="table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Class
                      </th>

                      <th>
                        Precision
                      </th>

                      <th>
                        Recall
                      </th>

                      <th>
                        F1 Score
                      </th>

                      <th>
                        Support
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {Object.entries(
                      evaluation.classification_report
                    ).map(
                      ([label, values]) => {

                        if (
                          typeof values !==
                          "object" ||
                          values === null
                        ) {

                          return null;

                        }


                        if (
                          !(
                            "precision" in values
                          )
                        ) {

                          return null;

                        }


                        return (

                          <tr key={label}>

                            <td>
                              {label}
                            </td>

                            <td>
                              {formatPercent(
                                values.precision
                              )}
                            </td>

                            <td>
                              {formatPercent(
                                values.recall
                              )}
                            </td>

                            <td>
                              {formatPercent(
                                values["f1-score"]
                              )}
                            </td>

                            <td>
                              {values.support}
                            </td>

                          </tr>

                        );

                      }
                    )}

                  </tbody>

                </table>

              </div>

            </div>

          )}


          {/* ==================================================
              CONFUSION MATRIX
              ================================================== */}

          {evaluation.confusion_matrix && (

            <div className="evaluation-panel">

              <h3>
                Confusion Matrix
              </h3>


              <div className="table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Actual / Predicted
                      </th>

                      <th>
                        Negative
                      </th>

                      <th>
                        Neutral
                      </th>

                      <th>
                        Positive
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {evaluation.confusion_matrix.map(
                      (row, index) => (

                        <tr key={index}>

                          <td>
                            {
                              [
                                "Negative",
                                "Neutral",
                                "Positive"
                              ][index]
                            }
                          </td>

                          <td>
                            {row[0]}
                          </td>

                          <td>
                            {row[1]}
                          </td>

                          <td>
                            {row[2]}
                          </td>

                        </tr>

                      )
                    )}

                  </tbody>

                </table>

              </div>

            </div>

          )}


          {/* ==================================================
              ROC CURVE
              ================================================== */}

          {evaluation.roc_auc && (

            <div className="evaluation-panel">

              <h3>
                ROC Curve
              </h3>

              <p className="evaluation-description">

                One-vs-rest ROC curves for
                Negative, Neutral and Positive
                sentiment classes.

              </p>


              <div className="roc-container">

                {drawRoc(
                  evaluation.roc
                )}

              </div>


              <div className="roc-legend">

                <div>

                  <span className="legend-dot negative-dot"></span>

                  Negative

                  <strong>
                    AUC:{" "}
                    {Number(
                      evaluation.roc_auc.negative
                    ).toFixed(4)}
                  </strong>

                </div>


                <div>

                  <span className="legend-dot neutral-dot"></span>

                  Neutral

                  <strong>
                    AUC:{" "}
                    {Number(
                      evaluation.roc_auc.neutral
                    ).toFixed(4)}
                  </strong>

                </div>


                <div>

                  <span className="legend-dot positive-dot"></span>

                  Positive

                  <strong>
                    AUC:{" "}
                    {Number(
                      evaluation.roc_auc.positive
                    ).toFixed(4)}
                  </strong>

                </div>

              </div>

            </div>

          )}

        </div>

      )}

    </section>

  );

}


export default ModelEvaluation;