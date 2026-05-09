// Animation loop and neuron animations
import { CONFIG } from './config.js';
import { state } from './state.js';

export function startAnimation() {
    animate();
}

function animate() {
    state.animationId = requestAnimationFrame(animate);
    state.time += 0.05;

    // Animate neurons
    animateNeurons();

    // Render scene
    state.renderer.render(state.scene, state.camera);
}

function animateNeurons() {
    const { pulseSpeed, pulseAmount, interpolationSpeed } = CONFIG.animation;
    const maxScale = 1.0 + pulseAmount; // Cap the maximum scale
    
    if (state.isRunning) {
        state.neurons.flat().forEach((neuron) => {
            if (neuron.userData.isActive) {
                // Smooth pulsing for active neurons
                const pulse = Math.sin(state.time * pulseSpeed) * pulseAmount + (1 + pulseAmount);
                neuron.userData.targetScale = Math.min(pulse, maxScale);
            } else {
                neuron.userData.targetScale = 1.0;
            }
            
            // Smooth scale interpolation
            const currentScale = neuron.scale.x;
            const newScale = currentScale + (neuron.userData.targetScale - currentScale) * interpolationSpeed;
            neuron.scale.set(newScale, newScale, newScale);
        });
    } else {
        // When stopped, return all neurons to base state
        state.neurons.flat().forEach((neuron) => {
            const currentScale = neuron.scale.x;
            const newScale = currentScale + (1.0 - currentScale) * 0.1;
            neuron.scale.set(newScale, newScale, newScale);
        });
    }
}

export function stopAnimation() {
    if (state.animationId) {
        cancelAnimationFrame(state.animationId);
        state.animationId = null;
    }
}