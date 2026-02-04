// Mouse and camera controls
import { CONFIG } from './config.js';
import { state } from './state.js';

export function updateCameraPosition() {
    const { cameraDistance, cameraRotation, camera } = state;
    camera.position.x = cameraDistance * Math.sin(cameraRotation.y) * Math.cos(cameraRotation.x);
    camera.position.y = cameraDistance * Math.sin(cameraRotation.x);
    camera.position.z = cameraDistance * Math.cos(cameraRotation.y) * Math.cos(cameraRotation.x);
    camera.lookAt(0, 0, 0);
}

export function initControls() {
    const container = document.getElementById('canvas-container');
    
    container.addEventListener('mousedown', onMouseDown);
    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('mouseup', onMouseUp);
    container.addEventListener('mouseleave', onMouseUp);
    container.addEventListener('wheel', onWheel);
}

function onMouseDown(e) {
    state.isDragging = true;
    state.previousMousePosition = { x: e.clientX, y: e.clientY };
}

function onMouseMove(e) {
    if (!state.isDragging) return;
    
    const { rotationSpeed } = CONFIG.controls;
    const deltaX = e.clientX - state.previousMousePosition.x;
    const deltaY = e.clientY - state.previousMousePosition.y;

    state.cameraRotation.y += deltaX * rotationSpeed;
    state.cameraRotation.x += deltaY * rotationSpeed;

    // Limit vertical rotation
    state.cameraRotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, state.cameraRotation.x));

    updateCameraPosition();
    state.previousMousePosition = { x: e.clientX, y: e.clientY };
}

function onMouseUp() {
    state.isDragging = false;
}

function onWheel(e) {
    e.preventDefault();
    
    const { zoomSpeed } = CONFIG.controls;
    const { minDistance, maxDistance } = CONFIG.camera;
    const delta = e.deltaY > 0 ? 1 : -1;
    const newDistance = state.cameraDistance + delta * zoomSpeed * state.cameraDistance;
    
    if (newDistance > minDistance && newDistance < maxDistance) {
        state.cameraDistance = newDistance;
        updateCameraPosition();
    }
}