// Neural network creation and management
import { CONFIG } from './config.js';
import { state } from './state.js';

export function createNetwork() {
    const { layers, layerSpacing, verticalSpacing, neuronSize, neuronSegments } = CONFIG.network;
    const totalLayers = layers.length;

    layers.forEach((neuronCount, layerIdx) => {
        const layerNeurons = [];
        const x = (layerIdx - (totalLayers - 1) / 2) * layerSpacing;
        const yOffset = ((neuronCount - 1) * verticalSpacing) / 2;

        for (let i = 0; i < neuronCount; i++) {
            const y = i * verticalSpacing - yOffset;
            const neuron = createNeuron(x, y, 0);
            layerNeurons.push(neuron);
        }

        state.neurons.push(layerNeurons);

        // Create connections to previous layer
        if (layerIdx > 0) {
            createConnections(state.neurons[layerIdx - 1], layerNeurons);
        }
    });
}

function createNeuron(x, y, z) {
    const { neuronSize, neuronSegments } = CONFIG.network;
    const { neuronBase } = CONFIG.colors;
    const { metalness, roughness, emissiveIntensity } = CONFIG.materials.neuron;

    // Main neuron sphere
    const geometry = new THREE.SphereGeometry(neuronSize, neuronSegments, neuronSegments);
    const material = new THREE.MeshStandardMaterial({
        color: neuronBase,
        metalness: metalness,
        roughness: roughness,
        emissive: 0x000000,
        emissiveIntensity: emissiveIntensity
    });
    
    const neuron = new THREE.Mesh(geometry, material);
    neuron.position.set(x, y, z);
    neuron.castShadow = true;
    neuron.receiveShadow = true;
    state.scene.add(neuron);
    
    // Store neuron state
    neuron.userData = {
        baseColor: CONFIG.colors.neuronBase,
        activeColor: CONFIG.colors.neuronActive,
        baseEmissive: 0x000000,
        activeEmissive: CONFIG.colors.neuronEmissiveActive,
        isActive: false,
        targetScale: 1.0
    };
    
    return neuron;
}
function createConnections(prevLayer, currentLayer) {
    const { opacityMin, opacityMax } = CONFIG.materials.connection;
    const connectionColors = [
        CONFIG.colors.connectionWeak,
        CONFIG.colors.connectionMedium,
        CONFIG.colors.connectionStrong,
        CONFIG.colors.connectionVeryStrong
    ];

    currentLayer.forEach((neuron) => {
        prevLayer.forEach((prevNeuron) => {
            // Randomly assign connection strength/color
            const strength = Math.random();
            let color;
            if (strength < 0.3) color = connectionColors[0];
            else if (strength < 0.6) color = connectionColors[1];
            else if (strength < 0.85) color = connectionColors[2];
            else color = connectionColors[3];

            // Create curved connection using quadratic bezier curve
            const start = prevNeuron.position;
            const end = neuron.position;
            
            // Calculate control point for curve
            const midPoint = new THREE.Vector3(
                (start.x + end.x) / 2,
                (start.y + end.y) / 2,
                (start.z + end.z) / 2
            );
            
            // Add some randomness to create varied curves
            const curvature = (Math.random() - 0.5) * 3;
            const controlPoint = new THREE.Vector3(
                midPoint.x + curvature,
                midPoint.y + curvature * 0.5,
                midPoint.z + curvature
            );
            
            // Create curved path
            const curve = new THREE.QuadraticBezierCurve3(start, controlPoint, end);
            const points = curve.getPoints(20);
            
            // Use TubeGeometry for thicker lines
            const tubeGeometry = new THREE.TubeGeometry(curve, 20, 0.02, 8, false);
            const material = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: opacityMin + (opacityMax - opacityMin) * strength,
            });
            const tube = new THREE.Mesh(tubeGeometry, material);
            state.scene.add(tube);
            state.connections.push({ 
                line: tube, 
                material, 
                baseOpacity: opacityMin + (opacityMax - opacityMin) * strength,
                strength: strength 
            });
        });
    });
}

// function createConnections(prevLayer, currentLayer) {
//     const { connectionBase } = CONFIG.colors;
//     const { opacity } = CONFIG.materials.connection;

//     currentLayer.forEach((neuron) => {
//         prevLayer.forEach((prevNeuron) => {
//             const points = [prevNeuron.position, neuron.position];
//             const geometry = new THREE.BufferGeometry().setFromPoints(points);
//             const material = new THREE.LineBasicMaterial({
//                 color: connectionBase,
//                 transparent: true,
//                 opacity: opacity,
//                 linewidth: 1
//             });
//             const line = new THREE.Line(geometry, material);
//             state.scene.add(line);
//             state.connections.push({ line, material });
//         });
//     });
// }

export function updateNeuronActivation(neurons, tpmData) {
    const { tpmThreshold } = CONFIG.animation;
    const { activeColor, neuronEmissiveActive, neuronBase } = CONFIG.colors;
    
    neurons.flat().forEach((neuron, idx) => {
        if (tpmData.length > 0 && tpmData[idx]) {
            const activity = tpmData[idx].reduce((a, b) => a + b, 0) / tpmData[idx].length;
            
            if (activity > tpmThreshold) {
                neuron.userData.isActive = true;
                neuron.material.color.setHex(activeColor);
                neuron.material.emissive.setHex(neuronEmissiveActive);
                neuron.material.emissiveIntensity = 0.5;
            } else {
                neuron.userData.isActive = false;
                neuron.material.color.setHex(neuronBase);
                neuron.material.emissive.setHex(neuron.userData.baseEmissive);
                neuron.material.emissiveIntensity = 0;
            }
        }
    });
}

export function resetNeurons(neurons) {
    neurons.flat().forEach((neuron) => {
        neuron.userData.isActive = false;
        neuron.material.color.setHex(neuron.userData.baseColor);
        neuron.material.emissive.setHex(neuron.userData.baseEmissive);
        neuron.material.emissiveIntensity = 0;
        neuron.scale.set(1, 1, 1);
    });
}