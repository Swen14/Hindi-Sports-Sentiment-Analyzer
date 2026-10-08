import { useEffect, useState } from "react";
import {
  Cpu,
  Database,
  FlaskConical,
  LineChart,
  SlidersHorizontal,
} from "lucide-react";

import { GROUP_STYLE, fetchEvaluation, percent } from "../api";
import ModelPicker from "./ModelPicker";
import EvaluationView from "./EvaluationView";


function Spec({ label, value, mono = false }) {
  return (
    <div className="spec">
      <span>{label}</span>
      <strong className={mono ? "mono" : ""}>{value ?? "—"}</strong>
    </div>
  );
}


function HistoryChart({ history }) {
  if (!history?.length) {
    return <p className="muted small">No validation history saved for this model.</p>;
  }
  const min = Math.min(...history.map((h) => h.val_macro_f1));
  const floor = Math.max(0, Math.floor((min - 0.05) * 20) / 20);

  return (
    <div className="history">
      {history.map((h) => {
        const height = ((h.val_macro_f1 - floor) / (1 - floor || 1)) * 100;
        return (
          <div className="history-col" key={h.epoch} title={`Val loss ${h.val_loss}`}>
            <span className="history-value">{percent(h.val_macro_f1)}</span>
            <div className="history-bar-track">
              <div className="history-bar" style={{ height: `${Math.max(height, 4)}%` }} />
            </div>
            <span className="history-epoch">Epoch {h.epoch}</span>
          </div>
        );
      })}
    </div>
  );
}


function ModelLab({ models, groups, testSets }) {

  const [selected, setSelected] = useState("real_muril");
  const [testSet, setTestSet] = useState("own");
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const model = models.find((m) => m.id === selected);

  useEffect(() => {
    if (!model?.available) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    fetchEvaluation(selected, testSet)
      .then((data) => !cancelled && setEvaluation(data))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [selected, testSet, model?.available]);

  const ownSet = model?.own_test_set;
  const otherSet = ownSet === "real" ? "synthetic" : "real";
  const style = model ? GROUP_STYLE[model.group] : null;
  const training = model?.training;
  const hp = training?.hyperparameters || {};

  return (
    <section id="lab" className="section">

      <div className="section-head">
        <span className="eyebrow"><FlaskConical size={14} /> Model lab</span>
        <h2>Pick a model to inspect</h2>
        <p>
          Each architecture was fine-tuned twice. Choose a model to see its
          architecture, training setup and full evaluation.
        </p>
      </div>

      <ModelPicker
        models={models}
        groups={groups}
        selected={selected}
        onSelect={(id) => { setSelected(id); setTestSet("own"); }}
      />

      {model && (
        <div className={`card details tone-${style.tone}`}>

          <div className="details-head">
            <div>
              <span className={`tag tag-${style.tone}`}>Trained on {style.label.toLowerCase()}</span>
              <h3>{model.name}</h3>
              <code>{model.architecture.base_model}</code>
            </div>
            <div className="details-scores">
              <div>
                <span>Own test F1</span>
                <strong>{percent(model.scores?.[ownSet]?.macro_f1)}</strong>
              </div>
              <div>
                <span>Real-world F1</span>
                <strong>{percent(model.scores?.real?.macro_f1)}</strong>
              </div>
            </div>
          </div>

          <div className="spec-grid">

            <div className="spec-card">
              <h4><Cpu size={16} /> Architecture</h4>
              <Spec label="Backbone" value={model.architecture.architecture} />
              <Spec label="Pre-training" value={model.architecture.pretraining} />
              <Spec label="Tokenizer" value={model.architecture.tokenizer} />
              <Spec
                label="Parameters"
                value={training.parameters
                  ? `${(training.parameters / 1e6).toFixed(1)} M`
                  : null}
              />
              <Spec label="Output classes" value="3 (Negative / Neutral / Positive)" />
            </div>

            <div className="spec-card">
              <h4><Database size={16} /> Training data</h4>
              <Spec label="Dataset" value={training.dataset} mono />
              <Spec label="Train samples" value={training.train_size?.toLocaleString()} />
              <Spec label="Validation samples" value={training.validation_size?.toLocaleString()} />
              <Spec label="Test samples" value={training.test_size?.toLocaleString()} />
              <Spec label="Source" value={style.short} />
            </div>

            <div className="spec-card">
              <h4><SlidersHorizontal size={16} /> Hyperparameters</h4>
              <Spec label="Epochs" value={hp.epochs} />
              <Spec label="Learning rate" value={hp.learning_rate?.toExponential()} mono />
              <Spec label="Batch size (train / eval)" value={`${hp.train_batch_size} / ${hp.eval_batch_size}`} />
              <Spec label="Weight decay" value={hp.weight_decay} />
              <Spec label="Max sequence length" value={hp.max_length} />
              <Spec label="Optimizer" value="AdamW, linear decay, fp16" />
              <Spec
                label="Training time"
                value={training.training_minutes ? `${training.training_minutes} min (RTX 3050)` : null}
              />
            </div>

            <div className="spec-card">
              <h4><LineChart size={16} /> Validation macro F1</h4>
              <HistoryChart history={training.history} />
              <p className="muted small">Best epoch is kept (model selection on validation macro F1).</p>
            </div>

          </div>

          <div className="eval-switch">
            <span>Evaluate on</span>
            <div className="segmented">
              <button
                className={testSet === "own" ? "active" : ""}
                onClick={() => setTestSet("own")}
              >
                Own test set · {testSets?.[ownSet]?.name}
              </button>
              <button
                className={testSet === otherSet ? "active" : ""}
                onClick={() => setTestSet(otherSet)}
              >
                Cross-domain · {testSets?.[otherSet]?.name}
              </button>
            </div>
          </div>

          <EvaluationView
            evaluation={evaluation}
            loading={loading}
            error={error}
            available={model.available}
          />
        </div>
      )}
    </section>
  );
}


export default ModelLab;
