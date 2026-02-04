// Main initialization
import { CONFIG } from './config.js';
import { state } from './state.js';
import { generateTPM, updateTPMDisplay } from './tpm.js';
import { initScene, onWindowResize } from './scene.js';
import { createNetwork } from './network.js';
import { initControls } from './controls.js';
import { startAnimation } from './animation.js';
import { initUI } from './ui.js';

function init() {
    // Initialize scene
    initScene();
    
    // Create neural network
    createNetwork();
    
    // Initialize controls
    initControls();
    
    // Initialize UI
    initUI();
    
    // Generate initial TPM
    const totalNeurons = CONFIG.network.layers.reduce((a, b) => a + b, 0);
    state.tpmData = generateTPM(totalNeurons);
    updateTPMDisplay(state.tpmData);
    
    // Start animation loop
    startAnimation();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}