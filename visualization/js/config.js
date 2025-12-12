// Configuration for the neural network
export const CONFIG = {
    network: {
        layers: [2, 8, 8, 1], // Network architecture: input, hidden, output
        layerSpacing: 5,
        verticalSpacing: 2,
        neuronSize: 0.4,
        neuronSegments: 64
    },
    
    camera: {
        fov: 60,
        near: 0.1,
        far: 1000,
        initialPosition: { x: 0, y: 3, z: 12},
        minDistance: 5,
        maxDistance: 30
    },
    
    colors: {
        background: 0x0f1419,
        neuronBase: 0xEFE9E3,
        neuronActive: 0x88b4e6,
        neuronEmissiveActive: 0x6699cc,
        connectionBase: 0x9ECFD4,
        connectionWeak: 0xff4444,      // Red
        connectionMedium: 0x44ff44,    // Green
        connectionStrong: 0xffaa00,    // Orange/Yellow
        connectionVeryStrong: 0x00ddff, // Cyan
        fog: 0x0f1419
    },
    
    materials: {
        neuron: {
            metalness: 0.2,
            roughness: 0.4,
            emissiveIntensity: 1
        },
        connection: {
            opacity: 0.7
        }
    },
    
    lighting: {
        ambient: { color: 0xffffff, intensity: 0.6 },
        main: { color: 0xffffff, intensity: 0.8, position: [5, 10, 7] },
        fill: { color: 0x6ea8ff, intensity: 0.3, position: [-5, 0, -5] },
        back: { color: 0x88ccff, intensity: 0.2, position: [0, -5, -10] }
    },
    
    animation: {
        pulseSpeed: 5,
        pulseAmount: 0.08,
        interpolationSpeed: 0.15,
        tpmThreshold: 0.5,
        simulationInterval: 500 // milliseconds
    },
    
    controls: {
        rotationSpeed: 0.005,
        zoomSpeed: 0.05
    }
};