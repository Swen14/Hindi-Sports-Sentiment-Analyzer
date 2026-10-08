import { AlertTriangle, LoaderCircle } from "lucide-react";

import { SENTIMENTS, percent } from "../api";


const ROC_COLORS = {
  negative: "var(--negative)",
  neutral: "var(--neutral)",
  positive: "var(--positive)",
};


function RocChart({ roc }) {
  const size = 300;
  const pad = 36;
  const inner = size - pad * 2;
  const toPoints = (points) =>
    points
      .map((p) => `${pad + p.fpr * inner},${pad + (1 - p.tpr) * inner}`)
      .join(" ");

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="roc">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={pad} x2={size - pad} y1={pad + t * inner} y2={pad + t * inner} className="roc-grid" />
          <line y1={pad} y2={size - pad} x1={pad + t * inner} x2={pad + t * inner} className="roc-grid" />
        </g>
      ))}
      <line x1={pad} y1={size - pad} x2={size - pad} y2={pad} className="roc-chance" />
      {Object.entries(roc).map(([label, points]) => (
        <polyline
          key={label}
          points={toPoints(points)}
          fill="none"
          stroke={ROC_COLORS[label]}
          strokeWidth="2.5"
        />
      ))}
      <text x={size / 2} y={size - 8} className="roc-axis-label">False positive rate</text>
      <text x={12} y={size / 2} className="roc-axis-label" transform={`rotate(-90 12 ${size / 2})`}>
        True positive rate
      </text>
    </svg>
  );
}


function ConfusionMatrix({ matrix }) {
  return (
    <div className="cm">
      <div className="cm-corner">Actual ↓ / Predicted →</div>
      {SENTIMENTS.map((label) => (
        <div className="cm-col-head" key={label}>{label}</div>
      ))}
      {matrix.map((row, i) => {
        const total = row.reduce((a, b) => a + b, 0) || 1;
        return [
          <div className="cm-row-head" key={`h${i}`}>{SENTIMENTS[i]}</div>,
          ...row.map((count, j) => {
            const share = count / total;
            return (
              <div
                key={`${i}-${j}`}
                className={i === j ? "cm-cell diag" : "cm-cell"}
                style={{ "--share": share }}
                title={`${(share * 100).toFixed(1)}% of actual ${SENTIMENTS[i]}`}
              >
                <strong>{count}</strong>
                <small>{(share * 100).toFixed(0)}%</small>
              </div>
            );
          }),
        ];
      })}
    </div>
  );
}


function EvaluationView({ evaluation, loading, error, available }) {

  if (!available) {
    return (
      <div className="banner banner-warn">
        <AlertTriangle size={16} />
        Model weights were not found on this machine. Train the model first.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="eval-loading">
        <LoaderCircle size={20} className="spin" />
        Running evaluation (cached after the first run)…
      </div>
    );
  }

  if (error) return <div className="banner banner-error">{error}</div>;
  if (!evaluation) return null;

  const report = evaluation.classification_report;

  return (
    <div className="evaluation">

      {evaluation.cross_domain && (
        <div className="banner banner-info">
          Cross-domain test: this model never saw data from the {evaluation.dataset.toLowerCase()} during training.
        </div>
      )}

      <div className="metrics">
        {[
          ["Accuracy", evaluation.accuracy],
          ["Macro precision", evaluation.macro_precision],
          ["Macro recall", evaluation.macro_recall],
          ["Macro F1", evaluation.macro_f1],
        ].map(([label, value]) => (
          <div className={label === "Macro F1" ? "metric highlight" : "metric"} key={label}>
            <span>{label}</span>
            <strong>{percent(value, 2)}</strong>
            <div className="metric-bar"><div style={{ width: `${value * 100}%` }} /></div>
          </div>
        ))}
      </div>

      <p className="muted small">
        {evaluation.dataset} · {evaluation.test_samples} comments
      </p>

      <div className="eval-grid">

        <div className="panel">
          <h4>Per-class metrics</h4>
          <table className="table">
            <thead>
              <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
            </thead>
            <tbody>
              {SENTIMENTS.map((label) => (
                <tr key={label}>
                  <td><span className={`dot dot-${label.toLowerCase()}`} />{label}</td>
                  <td>{percent(report[label].precision)}</td>
                  <td>{percent(report[label].recall)}</td>
                  <td><b>{percent(report[label]["f1-score"])}</b></td>
                  <td>{report[label].support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel">
          <h4>Confusion matrix</h4>
          <ConfusionMatrix matrix={evaluation.confusion_matrix} />
        </div>

        <div className="panel">
          <h4>ROC curves (one-vs-rest)</h4>
          <RocChart roc={evaluation.roc} />
          <div className="legend">
            {SENTIMENTS.map((label) => (
              <span key={label}>
                <i style={{ background: ROC_COLORS[label.toLowerCase()] }} />
                {label} <b>AUC {evaluation.roc_auc[label.toLowerCase()].toFixed(3)}</b>
              </span>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}


export default EvaluationView;
