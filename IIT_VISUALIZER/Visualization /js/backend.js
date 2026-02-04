/**
 * Backend Module - Communication with Python Flask server

 */

const BACKEND_URL = "http://localhost:5002";
let socket = null;
let isConnected = false;

// WebSocket event handlers
const eventHandlers = {
  onEpochUpdate: null,
  onConnect: null,
  onDisconnect: null,
  onError: null,
};

/**
 * Initialize WebSocket connection
 */
export function initWebSocket() {
  if (socket && socket.connected) {
    console.log("WebSocket already connected");
    return;
  }

  // Load Socket.IO client from CDN if not already loaded
  if (typeof io === "undefined") {
    const script = document.createElement("script");
    script.src =
      "https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js";
    script.onload = () => connectWebSocket();
    document.head.appendChild(script);
  } else {
    connectWebSocket();
  }
}

function connectWebSocket() {
  socket = io(BACKEND_URL, {
    transports: ["websocket", "polling"],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5,
  });

  socket.on("connect", () => {
    console.log("WebSocket connected");
    isConnected = true;
    if (eventHandlers.onConnect) {
      eventHandlers.onConnect();
    }
  });

  socket.on("disconnect", () => {
    console.log("WebSocket disconnected");
    isConnected = false;
    if (eventHandlers.onDisconnect) {
      eventHandlers.onDisconnect();
    }
  });

  socket.on("epoch_update", (data) => {
    console.log("Received epoch update:", data);
    if (eventHandlers.onEpochUpdate) {
      eventHandlers.onEpochUpdate(data);
    }
  });

  socket.on("error", (error) => {
    console.error("WebSocket error:", error);
    if (eventHandlers.onError) {
      eventHandlers.onError(error);
    }
  });

  socket.on("connection_response", (data) => {
    console.log("Connection response:", data);
  });
}

/**
 * Register event handler for epoch updates
 */
export function onEpochUpdate(callback) {
  eventHandlers.onEpochUpdate = callback;
}

/**
 * Register event handler for connection
 */
export function onConnect(callback) {
  eventHandlers.onConnect = callback;
}

/**
 * Register event handler for disconnection
 */
export function onDisconnect(callback) {
  eventHandlers.onDisconnect = callback;
}

/**
 * Register event handler for errors
 */
export function onError(callback) {
  eventHandlers.onError = callback;
}

/**
 * Check if WebSocket is connected
 */
export function isSocketConnected() {
  return isConnected && socket && socket.connected;
}

/**
 * Request latest update from server
 */
export function requestUpdate() {
  if (socket && socket.connected) {
    socket.emit("request_update");
  }
}

/**
 * Health check endpoint
 */
export async function healthCheck() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    const data = await response.json();
    console.log("Backend health:", data);
    return data;
  } catch (error) {
    console.error("Health check failed:", error);
    return null;
  }
}

/**
 * Get available models from backend
 */
export async function getAvailableModels() {
  try {
    const response = await fetch(`${BACKEND_URL}/models`);
    const data = await response.json();
    return data.models || [];
  } catch (error) {
    console.error("Failed to get models:", error);
    return ["MLP"];
  }
}

/**
 * Start model training
 * @param {string} modelName - Name of model to train
 * @param {number} epochs - Number of epochs
 */
export async function startTraining(modelName = "MLP", epochs = 10) {
  try {
    const response = await fetch(`${BACKEND_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name: modelName,
        epochs: epochs,
      }),
    });

    const data = await response.json();
    console.log("Training started:", data);
    return data;
  } catch (error) {
    console.error("Training failed:", error);
    return { success: false, error: error.message };
  }
}

/**
 * Compute Phi for a given TPM matrix
 * @param {Array} tpm - TPM matrix
 */
export async function computePhi(tpm) {
  try {
    const response = await fetch(`${BACKEND_URL}/compute_phi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tpm: tpm }),
    });

    const data = await response.json();
    return data.success ? data.phi : null;
  } catch (error) {
    console.error("Phi computation failed:", error);
    return null;
  }
}

/**
 * Compute minimum information partition (MIP)
 * @param {Array} tpm - TPM matrix
 */
export async function computeMinimumCut(tpm) {
  try {
    const response = await fetch(`${BACKEND_URL}/minimum_cut`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tpm: tpm }),
    });

    const data = await response.json();
    return data.success ? data.mip : null;
  } catch (error) {
    console.error("Minimum cut computation failed:", error);
    return null;
  }
}

/**
 * Get training history
 */
export async function getTrainingHistory() {
  try {
    const response = await fetch(`${BACKEND_URL}/history`);
    const data = await response.json();
    return data.success ? data.history : [];
  } catch (error) {
    console.error("Failed to get training history:", error);
    return [];
  }
}

// Legacy functions for backward compatibility
export async function initializeBackend(nSamples = 1000, seqLen = 6) {
  console.warn("initializeBackend is deprecated. Use healthCheck() instead.");
  return await healthCheck();
}

export async function trainEpochBackend() {
  console.warn("trainEpochBackend is deprecated. Use startTraining() instead.");
  return await startTraining("MLP", 1);
}

export async function getTPMFromBackend() {
  console.warn(
    "getTPMFromBackend is deprecated. Use getTrainingHistory() instead."
  );
  const history = await getTrainingHistory();
  return history.length > 0 ? { tpm: history[history.length - 1].tpm } : null;
}

export async function getNetworkStateFromBackend() {
  console.warn(
    "getNetworkStateFromBackend is deprecated. Use getTrainingHistory() instead."
  );
  const history = await getTrainingHistory();
  return history.length > 0
    ? { state: history[history.length - 1].network_state }
    : null;
}

export async function getModelInfo() {
  return await getAvailableModels();
}
