const API_URL = "http://127.0.0.1:5001/derivative";
const PREVIEW_URL = "http://127.0.0.1:5001/preview";

const input = document.getElementById("expression");
const button = document.getElementById("calcBtn");
const error = document.getElementById("error");
const visualization = document.getElementById("visualization");
const results = document.getElementById("results");

let previewTimer = null;
let previewRequestId = 0;

button.addEventListener("click", calculateDerivative);
input.addEventListener("input", updateLiveVisualization);
input.addEventListener("keydown", function (e) {
  if (e.key === "Enter") {
    calculateDerivative();
  }
});

updateLiveVisualization();

function typesetElements(elements) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise(elements);
  }
}

function updateLiveVisualization() {
  const expression = input.value.trim();

  if (previewTimer) {
    clearTimeout(previewTimer);
  }

  if (!expression) {
    visualization.innerHTML = "\\(x\\)";
    typesetElements([visualization]);
    return;
  }

  const currentRequestId = ++previewRequestId;

  previewTimer = setTimeout(async () => {
    try {
      const response = await fetch(PREVIEW_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ expression: expression })
      });

      const data = await response.json();

      if (currentRequestId !== previewRequestId) {
        return;
      }

      const previewLatex = data.latex || "x";
      visualization.innerHTML = "\\(" + previewLatex + "\\)";
      typesetElements([visualization]);
    } catch (err) {
      if (currentRequestId !== previewRequestId) {
        return;
      }
      visualization.innerHTML = "\\(" + expression + "\\)";
      typesetElements([visualization]);
    }
  }, 150);
}

function createResultCard(expressionLatex, derivativeLatex) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `
    <h2 class="result-card-title">Calculated derivative</h2>
    <div class="result-card-math">\\(${derivativeLatex}\\)</div>
  `;
  results.innerHTML = "";
  results.appendChild(card);
  typesetElements([card]);
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function calculateDerivative() {
  const expression = input.value.trim();
  let expressionToCalculate = expression;
  error.textContent = "";

  if (!expressionToCalculate) {
    expressionToCalculate = "sin(x)*cos(x)/(e^x*ln(x))";
    input.value = expressionToCalculate;
    updateLiveVisualization();
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ expression: expressionToCalculate })
    });

    const data = await response.json();

    if (!response.ok) {
      updateLiveVisualization();
      error.textContent = data.error || "Something went wrong.";
      return;
    }

    const expressionLatex = data.expression_latex || data.expression;
    const derivativeLatex = data.derivative_latex || data.derivative;

    visualization.innerHTML = "\\(\\frac{d}{dx}\\left[" + expressionLatex + "\\right]\\)";
    typesetElements([visualization]);
    createResultCard(expressionLatex, derivativeLatex);
  } catch (err) {
    updateLiveVisualization();
    error.textContent = "Could not connect to the server.";
  }
}