const API_BASE = (
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000"
).replace(/\/$/, "");


async function request(path, options) {

  let response;

  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error(
      "Could not connect to the backend. Start it with: uvicorn backend.main:app --reload"
    );
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }

  return data;
}


export function fetchModels() {
  return request("/models");
}


export function predict(text, model) {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
}


export function fetchEvaluation(model, testSet = "own") {
  return request(`/evaluation/${model}?test_set=${testSet}`);
}


export const SENTIMENTS = ["Negative", "Neutral", "Positive"];


export const GROUP_STYLE = {
  synthetic: { label: "Synthetic data", short: "AI-generated", tone: "violet" },
  real: { label: "Real-world data", short: "YouTube comments", tone: "teal" },
};


export function percent(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
