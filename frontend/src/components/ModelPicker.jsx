import { Bot, Globe2 } from "lucide-react";

import { GROUP_STYLE, percent } from "../api";


const GROUP_ICON = { synthetic: Bot, real: Globe2 };


function ModelPicker({ models, groups, selected, onSelect, compact = false }) {

  return (
    <div className={compact ? "picker compact" : "picker"}>
      {["synthetic", "real"].map((group) => {
        const Icon = GROUP_ICON[group];
        const style = GROUP_STYLE[group];
        const groupModels = models.filter((m) => m.group === group);

        return (
          <div className={`picker-group tone-${style.tone}`} key={group}>

            <div className="picker-group-head">
              <span className="picker-icon"><Icon size={compact ? 15 : 18} /></span>
              <div>
                <strong>
                  {groups?.[group]?.title || `Trained on ${style.label.toLowerCase()}`}
                </strong>
                {!compact && groups?.[group] && (
                  <p>{groups[group].description}</p>
                )}
              </div>
            </div>

            <div className="picker-buttons">
              {groupModels.map((model) => (
                <button
                  key={model.id}
                  className={selected === model.id ? "model-btn active" : "model-btn"}
                  onClick={() => onSelect(model.id)}
                  disabled={!model.available}
                  title={model.available ? model.architecture.base_model : "Model weights not found"}
                >
                  <span className="model-btn-name">{model.name}</span>
                  {!compact && (
                    <span className="model-btn-score">
                      {model.scores?.real
                        ? <>Real-world F1 <b>{percent(model.scores.real.macro_f1)}</b></>
                        : model.available ? "Not evaluated yet" : "Weights missing"}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}


export default ModelPicker;
