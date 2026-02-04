/**
 * Global State Management
 */

export const state = {
  // Simulation state
  epoch: 0,
  isRunning: false,
  currentPhi: 0,
  phiHistory: [],

  // Matrix data
  tpmData: [],
  mipData: [],

  // Node cut animation
  cutAnimation: {
    isActive: false,
    partition1: [],
    partition2: [],
    cutProgress: 0,
  },

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
  cameraDistance: 12,
};
