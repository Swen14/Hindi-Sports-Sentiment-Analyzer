import { BarChart3, Lightbulb } from "lucide-react";

import { GROUP_STYLE, percent } from "../api";


const TEST_SETS = [
  { id: "synthetic", label: "Synthetic test" },
  { id: "real", label: "Real-world test" },
];


function average(values) {
  const valid = values.filter((v) => typeof v === "number");
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
}


function Comparison({ models }) {

  const evaluated = models.filter((m) => m.scores?.real);

  const realF1 = (group) =>
    average(models.filter((m) => m.group === group).map((m) => m.scores?.real?.macro_f1));
  const syntheticF1 = (group) =>
    average(models.filter((m) => m.group === group).map((m) => m.scores?.synthetic?.macro_f1));

  const synOnReal = realF1("synthetic");
  const realOnReal = realF1("real");
  const best = [...evaluated].sort(
    (a, b) => b.scores.real.macro_f1 - a.scores.real.macro_f1
  )[0];

  return (
    <section id="compare" className="section">

      <div className="section-head">
        <span className="eyebrow"><BarChart3 size={14} /> Comparison</span>
        <h2>Synthetic vs real-world training</h2>
        <p>
          Every model is scored on both test sets. Macro F1 treats the three
          classes equally, so it isn't inflated by the many negative comments.
        </p>
      </div>

      {synOnReal !== null && realOnReal !== null && (
        <div className="insights">
          <div className="insight">
            <Lightbulb size={18} />
            <div>
              <strong>
                Models trained on real comments score {percent(realOnReal)} macro F1 on real-world text,
                versus {percent(synOnReal)} for models trained on synthetic data.
              </strong>
              <span>
                The synthetic models reach {percent(syntheticF1("synthetic"))} on their own template-based
                test set, but that score doesn't carry over to real fan comments.
              </span>
            </div>
          </div>
          {best && (
            <div className="insight">
              <BarChart3 size={18} />
              <div>
                <strong>Best on real-world text: {best.name} ({GROUP_STYLE[best.group].label.toLowerCase()})</strong>
                <span>Macro F1 {percent(best.scores.real.macro_f1)} · accuracy {percent(best.scores.real.accuracy)}</span>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="card">

        <div className="bars-legend">
          {TEST_SETS.map((t) => (
            <span key={t.id}><i className={`swatch swatch-${t.id}`} />{t.label}</span>
          ))}
        </div>

        <div className="bars">
          {models.map((model) => (
            <div className="bar-group" key={model.id}>
              <div className="bar-pair">
                {TEST_SETS.map((t) => {
                  const value = model.scores?.[t.id]?.macro_f1;
                  return (
                    <div className="bar-slot" key={t.id}>
                      <span className="bar-value">{value !== undefined ? percent(value, 0) : "—"}</span>
                      <div
                        className={`bar bar-${t.id}`}
                        style={{ height: `${(value ?? 0) * 100}%` }}
                        title={`${t.label}: ${percent(value, 2)}`}
                      />
                    </div>
                  );
                })}
              </div>
              <div className="bar-label">
                <strong>{model.name}</strong>
                <span className={`tag tag-${GROUP_STYLE[model.group].tone}`}>
                  {GROUP_STYLE[model.group].short}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Trained on</th>
                <th>Parameters</th>
                <th>Synthetic acc.</th>
                <th>Synthetic F1</th>
                <th>Real-world acc.</th>
                <th>Real-world F1</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.id} className={best?.id === model.id ? "best-row" : ""}>
                  <td><b>{model.name}</b></td>
                  <td>
                    <span className={`tag tag-${GROUP_STYLE[model.group].tone}`}>
                      {GROUP_STYLE[model.group].label}
                    </span>
                  </td>
                  <td>
                    {model.training.parameters
                      ? `${(model.training.parameters / 1e6).toFixed(0)} M`
                      : "—"}
                  </td>
                  <td>{percent(model.scores?.synthetic?.accuracy)}</td>
                  <td>{percent(model.scores?.synthetic?.macro_f1)}</td>
                  <td>{percent(model.scores?.real?.accuracy)}</td>
                  <td><b>{percent(model.scores?.real?.macro_f1)}</b></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}


export default Comparison;
