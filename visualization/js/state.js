// Global state management
export const state = {
    epoch: 0,
    isRunning: false,
    tpmData: [],
    
    // Three.js objects
    scene: null,
    camera: null,
    renderer: null,
    neurons: [],
    connections: [],
    
    // Animation
    animationId: null,
    simulationInterval: null,
    time: 0,
    
    // Camera controls
    isDragging: false,
    previousMousePosition: { x: 0, y: 0 },
    cameraRotation: { x: 0, y: 0 },
    cameraDistance: 12
};