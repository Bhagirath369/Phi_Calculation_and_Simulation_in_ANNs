// UI event handlers and updates
import { CONFIG } from './config.js';
import { state } from './state.js';
import { generateTPM, updateTPMDisplay } from './tpm.js';
import { updateNeuronActivation, resetNeurons } from './network.js';

export function initUI() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetBtn = document.getElementById('resetBtn');

    startBtn.addEventListener('click', startSimulation);
    stopBtn.addEventListener('click', stopSimulation);
    resetBtn.addEventListener('click', resetSimulation);
}

function startSimulation() {
    state.isRunning = true;
    
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Running';
    statusEl.className = 'value status-running';

    state.simulationInterval = setInterval(updateSimulation, CONFIG.animation.simulationInterval);
}

function stopSimulation() {
    state.isRunning = false;
    
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    
    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Stopped';
    statusEl.className = 'value status-stopped';

    if (state.simulationInterval) {
        clearInterval(state.simulationInterval);
    }
}

function resetSimulation() {
    stopSimulation();
    state.epoch = 0;
    document.getElementById('epochValue').textContent = state.epoch;

    const totalNeurons = CONFIG.network.layers.reduce((a, b) => a + b, 0);
    state.tpmData = generateTPM(totalNeurons);
    updateTPMDisplay(state.tpmData);
    
    resetNeurons(state.neurons);
}

function updateSimulation() {
    state.epoch++;
    document.getElementById('epochValue').textContent = state.epoch;

    const totalNeurons = CONFIG.network.layers.reduce((a, b) => a + b, 0);
    state.tpmData = generateTPM(totalNeurons);
    updateTPMDisplay(state.tpmData);

    // Update connection opacity
    state.connections.forEach((conn, idx) => {
        if (state.tpmData.length > 0) {
            const value = state.tpmData[idx % state.tpmData.length]?.[0] || 0;
            conn.material.opacity = 0.2 + value * 0.5;
        }
    });

    // Update neuron activation
    updateNeuronActivation(state.neurons, state.tpmData);
}