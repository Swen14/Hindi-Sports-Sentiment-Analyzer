import { useEffect, useState } from "react";

import {
  BarChart3,
  ChevronDown,
  CircleAlert,
  Crosshair,
  LoaderCircle,
  Radar,
  RefreshCw,
  Target,
  Trophy,
} from "lucide-react";


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
      "XLM-RoBERTa",

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


  useEffect(() => {

    loadEvaluation(selectedModel);

  }, []);


  const handleModelChange = (e) => {

    const model =
      e.target.value;

    setSelectedModel(model);

    loadEvaluation(model);

  };


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
  // ROC GRAPH
  // ============================================================

  const drawRoc = (rocData) => {

    if (!rocData) {
      return null;
    }


    const width = 700;
    const height = 400;

    const left = 70;
    const right = 25;
    const top = 25;
    const bottom = 55;

    const graphWidth =
      width - left - right;

    const graphHeight =
      height - top - bottom;


    const convertPoints = (points) => {

      if (
        !points ||
        points.length === 0
      ) {

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


        <line
          x1={left}
          y1={height - bottom}
          x2={width - right}
          y2={top}
          className="roc-random"
        />


        {rocData.negative && (

          <polyline
            points={convertPoints(
              rocData.negative
            )}
            className="roc-negative"
          />

        )}


        {rocData.neutral && (

          <polyline
            points={convertPoints(
              rocData.neutral
            )}
            className="roc-neutral"
          />

        )}


        {rocData.positive && (

          <polyline
            points={convertPoints(
              rocData.positive
            )}
            className="roc-positive"
          />

        )}


        <text
          x={width / 2}
          y={height - 12}
          textAnchor="middle"
          className="roc-label"
        >
          False Positive Rate
        </text>


        <text
          x="17"
          y={height / 2}
          textAnchor="middle"
          transform={
            `rotate(-90 17 ${height / 2})`
          }
          className="roc-label"
        >
          True Positive Rate
        </text>


        <text
          x={left - 4}
          y={height - bottom + 20}
          className="roc-tick"
        >
          0
        </text>


        <text
          x={width - right - 4}
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


  return (

    <div className="evaluation-wrapper">


      {/* ======================================================
          HEADER
          ====================================================== */}

      <div className="evaluation-top">

        <div>

          <div className="section-kicker">

            <BarChart3 size={14} />

            MODEL EVALUATION

          </div>

          <h2>
            Model Performance
          </h2>

          <p>
            Compare transformer models using
            accuracy, precision, recall, F1,
            confusion matrix and ROC-AUC.
          </p>

        </div>


        <div className="evaluation-live">

          <span></span>

          BENCHMARK ACTIVE

        </div>

      </div>


      {/* ======================================================
          MODEL SELECTOR
          ====================================================== */}

      <div className="evaluation-selector">

        <div>

          <span>
            SELECT LANGUAGE MODEL
          </span>

          <strong>
            {modelNames[selectedModel]}
          </strong>

        </div>


        <div className="evaluation-select-wrapper">

          <Target size={17} />

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

          <ChevronDown
            size={16}
          />

        </div>


        <button
          className="refresh-button"
          onClick={() =>
            loadEvaluation(selectedModel)
          }
          disabled={loading}
          title="Refresh evaluation"
        >

          <RefreshCw
            size={17}
            className={
              loading
                ? "spin"
                : ""
            }
          />

        </button>

      </div>


      {/* ======================================================
          LOADING
          ====================================================== */}

      {loading && (

        <div className="evaluation-loading">

          <LoaderCircle
            size={22}
            className="spin"
          />

          <div>

            <strong>
              Loading evaluation
            </strong>

            <span>
              Running benchmark for{" "}
              {modelNames[selectedModel]}
            </span>

          </div>

        </div>

      )}


      {/* ======================================================
          ERROR
          ====================================================== */}

      {error && !loading && (

        <div className="evaluation-error">

          <CircleAlert size={21} />

          <div>

            <strong>
              Evaluation unavailable
            </strong>

            <p>
              {error}
            </p>

          </div>

        </div>

      )}


      {/* ======================================================
          RESULTS
          ====================================================== */}

      {evaluation && !loading && (

        <div className="evaluation-results">


          {/* CURRENT MODEL */}

          <div className="selected-model-banner">

            <div className="selected-model-icon">

              <Trophy size={20} />

            </div>

            <div>

              <span>
                CURRENTLY SELECTED MODEL
              </span>

              <strong>
                {modelNames[selectedModel]}
              </strong>

            </div>

            <div className="benchmark-pill">

              <span></span>

              EVALUATED

            </div>

          </div>


          {/* ==================================================
              METRICS
              ================================================== */}

          <div className="metric-grid">


            <div className="metric-card">

              <div className="metric-icon">
                <Target size={18} />
              </div>

              <span>
                ACCURACY
              </span>

              <strong>
                {formatPercent(
                  evaluation.accuracy
                )}
              </strong>

            </div>


            <div className="metric-card">

              <div className="metric-icon">
                <Crosshair size={18} />
              </div>

              <span>
                MACRO PRECISION
              </span>

              <strong>
                {formatPercent(
                  evaluation.macro_precision
                )}
              </strong>

            </div>


            <div className="metric-card">

              <div className="metric-icon">
                <Radar size={18} />
              </div>

              <span>
                MACRO RECALL
              </span>

              <strong>
                {formatPercent(
                  evaluation.macro_recall
                )}
              </strong>

            </div>


            <div className="metric-card highlight">

              <div className="metric-icon">
                <Trophy size={18} />
              </div>

              <span>
                MACRO F1
              </span>

              <strong>
                {formatPercent(
                  evaluation.macro_f1
                )}
              </strong>

            </div>

          </div>


          {/* ==================================================
              CLASSIFICATION REPORT
              ================================================== */}

          {evaluation.classification_report && (

            <div className="evaluation-panel">

              <div className="evaluation-panel-header">

                <div>

                  <span>
                    DETAILED METRICS
                  </span>

                  <h3>
                    Classification Report
                  </h3>

                </div>

                <BarChart3 size={20} />

              </div>


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

              <div className="evaluation-panel-header">

                <div>

                  <span>
                    ERROR ANALYSIS
                  </span>

                  <h3>
                    Confusion Matrix
                  </h3>

                </div>

                <Crosshair size={20} />

              </div>


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
                                "Positive",
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

            <div className="evaluation-panel roc-panel">

              <div className="evaluation-panel-header">

                <div>

                  <span>
                    CLASSIFICATION ANALYSIS
                  </span>

                  <h3>
                    ROC Curve
                  </h3>

                  <p>
                    One-vs-rest ROC curves for
                    Negative, Neutral and Positive
                    sentiment classes.
                  </p>

                </div>

                <Radar size={20} />

              </div>


              <div className="roc-container">

                {drawRoc(
                  evaluation.roc
                )}

              </div>


              <div className="roc-legend">


                <div>

                  <span className="legend-dot negative-dot"></span>

                  <span>
                    Negative
                  </span>

                  <strong>
                    AUC{" "}
                    {Number(
                      evaluation.roc_auc.negative
                    ).toFixed(4)}
                  </strong>

                </div>


                <div>

                  <span className="legend-dot neutral-dot"></span>

                  <span>
                    Neutral
                  </span>

                  <strong>
                    AUC{" "}
                    {Number(
                      evaluation.roc_auc.neutral
                    ).toFixed(4)}
                  </strong>

                </div>


                <div>

                  <span className="legend-dot positive-dot"></span>

                  <span>
                    Positive
                  </span>

                  <strong>
                    AUC{" "}
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

    </div>

  );

}


export default ModelEvaluation;