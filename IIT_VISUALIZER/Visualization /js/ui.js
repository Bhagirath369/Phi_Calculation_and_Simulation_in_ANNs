/**
 * UI Module - Simplified with simulated Phi values
 */

import { CONFIG } from "./config.js";
import { state } from "./state.js";
import {
  generateTPM,
  generateMIPMatrix,
  updateTPMDisplay,
  updateMIPDisplay,
} from "./tpm.js";
import {
  updateNeuronActivation,
  resetNeurons,
  animateNodeCut,
} from "./network.js";

export function initUI() {
  setupButtonListeners();
  initializeDisplays();
}

function setupButtonListeners() {
  document
    .getElementById("startBtn")
    .addEventListener("click", startSimulation);
  document.getElementById("stopBtn").addEventListener("click", stopSimulation);
  document
    .getElementById("resetBtn")
    .addEventListener("click", resetSimulation);
}

function initializeDisplays() {
  const totalNeurons = CONFIG.network.layers.reduce((a, b) => a + b, 0);
  state.tpmData = generateTPM(totalNeurons);
  state.mipData = generateMIPMatrix(totalNeurons);
  updateTPMDisplay(state.tpmData);
  updateMIPDisplay(state.mipData);
}

function startSimulation() {
  state.isRunning = true;
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").disabled = false;

  const statusEl = document.getElementById("status");
  statusEl.textContent = "Running";
  statusEl.className = "value status-running";

  state.simulationInterval = setInterval(
    updateSimulation,
    CONFIG.animation.simulationInterval,
  );
}

function stopSimulation() {
  state.isRunning = false;
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;

  const statusEl = document.getElementById("status");
  statusEl.textContent = "Stopped";
  statusEl.className = "value status-stopped";

  if (state.simulationInterval) {
    clearInterval(state.simulationInterval);
  }
}

function resetSimulation() {
  stopSimulation();

  state.epoch = 0;
  state.phiHistory = [];
  state.currentPhi = 0;

  document.getElementById("epochValue").textContent = state.epoch;
  document.getElementById("phiValue").textContent = "0.0000";
  document.getElementById("chartPhiValue").textContent = "0.0000";
  document.getElementById("chartEpochValue").textContent = state.epoch;

  const totalNeurons = CONFIG.network.layers.reduce((a, b) => a + b, 0);
  state.tpmData = generateTPM(totalNeurons);
  state.mipData = generateMIPMatrix(totalNeurons);

  updateTPMDisplay(state.tpmData);
  updateMIPDisplay(state.mipData);
  resetNeurons(state.neurons);
  updatePhiChart();
}

function updateSimulation() {
  state.epoch++;
  document.getElementById("epochValue").textContent = state.epoch;

  // Generate simulated Phi value (0-2 range, gradually increasing with asymptote)
  const targetPhi = 1.8; // Asymptotic limit
  const growthRate = 0.15; // How quickly it approaches the limit
  const noise = (Math.random() - 0.5) * 0.1; // Small random variations

  // Logistic growth formula with noise
  const rawPhi = targetPhi * (1 - Math.exp(-growthRate * state.epoch)) + noise;
  state.currentPhi = Math.max(0, Math.min(2, rawPhi)); // Clamp between 0-2

  // Update Phi display
  document.getElementById("phiValue").textContent = state.currentPhi.toFixed(4);
  document.getElementById("chartPhiValue").textContent =
    state.currentPhi.toFixed(4);
  document.getElementById("chartEpochValue").textContent = state.epoch;
  state.phiHistory.push({ epoch: state.epoch, phi: state.currentPhi });

  // Generate new matrices
  const totalNeurons = CONFIG.network.layers.reduce((a, b) => a + b, 0);
  state.tpmData = generateTPM(totalNeurons);
  state.mipData = generateMIPMatrix(totalNeurons);

  updateTPMDisplay(state.tpmData);
  updateMIPDisplay(state.mipData);

  // Animate node cut visualization every few epochs
  if (state.epoch % 3 === 0) {
    animateNodeCut(state.neurons);
  }

  // Update connection opacity based on TPM
  state.connections.forEach((conn, idx) => {
    if (state.tpmData.length > 0) {
      const value = state.tpmData[idx % state.tpmData.length]?.[0] || 0;
      conn.material.opacity = 0.2 + value * 0.5;
    }
  });

  // Update neuron activation
  updateNeuronActivation(state.neurons, state.tpmData);

  // Update Phi chart
  updatePhiChart();
}

function updatePhiChart() {
  const chartContainer = document.getElementById("phiChart");
  if (!chartContainer || state.phiHistory.length === 0) return;

  const width = chartContainer.clientWidth;
  const height = chartContainer.clientHeight;
  const padding = 40;

  chartContainer.innerHTML = "";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);

  const maxPhi = 2; // Fixed scale 0-2
  const minPhi = 0;
  const maxEpoch = Math.max(
    state.phiHistory[state.phiHistory.length - 1].epoch,
    10,
  );

  const xScale = (epoch) =>
    padding + (epoch / maxEpoch) * (width - 2 * padding);
  const yScale = (phi) =>
    height -
    padding -
    ((phi - minPhi) / (maxPhi - minPhi)) * (height - 2 * padding);

  // Draw grid lines
  for (let i = 0; i <= 4; i++) {
    const y = yScale(i * 0.5);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padding);
    line.setAttribute("y1", y);
    line.setAttribute("x2", width - padding);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", "#2d3748");
    line.setAttribute("stroke-width", "1");
    line.setAttribute("stroke-dasharray", "2,2");
    svg.appendChild(line);

    const label = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    label.setAttribute("x", padding - 25);
    label.setAttribute("y", y + 4);
    label.setAttribute("fill", "#9ca3af");
    label.setAttribute("font-size", "10");
    label.textContent = (i * 0.5).toFixed(1);
    svg.appendChild(label);
  }

  // Draw line
  const points = state.phiHistory
    .map((d) => `${xScale(d.epoch)},${yScale(d.phi)}`)
    .join(" ");
  const polyline = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "polyline",
  );
  polyline.setAttribute("points", points);
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "#6366f1");
  polyline.setAttribute("stroke-width", "2");
  svg.appendChild(polyline);

  // Draw area under curve
  const areaPoints =
    `${padding},${yScale(0)} ` +
    points +
    ` ${xScale(state.phiHistory[state.phiHistory.length - 1].epoch)},${yScale(0)}`;
  const polygon = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "polygon",
  );
  polygon.setAttribute("points", areaPoints);
  polygon.setAttribute("fill", "rgba(99, 102, 241, 0.1)");
  svg.appendChild(polygon);

  // Draw points
  state.phiHistory.forEach((d, i) => {
    if (
      i % Math.max(1, Math.floor(state.phiHistory.length / 20)) === 0 ||
      i === state.phiHistory.length - 1
    ) {
      const circle = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle",
      );
      circle.setAttribute("cx", xScale(d.epoch));
      circle.setAttribute("cy", yScale(d.phi));
      circle.setAttribute("r", "3");
      circle.setAttribute("fill", "#6366f1");
      svg.appendChild(circle);
    }
  });

  // Current value label
  if (state.phiHistory.length > 0) {
    const latest = state.phiHistory[state.phiHistory.length - 1];
    const valueText = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    valueText.setAttribute("x", width - padding - 5);
    valueText.setAttribute("y", "15");
    valueText.setAttribute("fill", "#6366f1");
    valueText.setAttribute("font-size", "12");
    valueText.setAttribute("font-weight", "bold");
    valueText.setAttribute("text-anchor", "end");
    valueText.textContent = `Φ = ${latest.phi.toFixed(4)}`;
    svg.appendChild(valueText);
  }

  // Axis labels
  const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xLabel.setAttribute("x", width / 2);
  xLabel.setAttribute("y", height - 10);
  xLabel.setAttribute("fill", "#9ca3af");
  xLabel.setAttribute("font-size", "11");
  xLabel.setAttribute("text-anchor", "middle");
  xLabel.textContent = "Epoch";
  svg.appendChild(xLabel);

  const yLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yLabel.setAttribute("x", 15);
  yLabel.setAttribute("y", height / 2);
  yLabel.setAttribute("fill", "#9ca3af");
  yLabel.setAttribute("font-size", "11");
  yLabel.setAttribute("text-anchor", "middle");
  yLabel.setAttribute("transform", `rotate(-90, 15, ${height / 2})`);
  yLabel.textContent = "Φ (Phi)";
  svg.appendChild(yLabel);

  chartContainer.appendChild(svg);
}
