// Scene setup and initialization
import { CONFIG } from './config.js';
import { state } from './state.js';

export function initScene() {
    const container = document.getElementById('canvas-container');

    // Scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(CONFIG.colors.background);
    state.scene.fog = new THREE.Fog(CONFIG.colors.fog, 10, 50);

    // Camera
    state.camera = new THREE.PerspectiveCamera(
        CONFIG.camera.fov,
        container.clientWidth / container.clientHeight,
        CONFIG.camera.near,
        CONFIG.camera.far
    );
    state.camera.position.set(
        CONFIG.camera.initialPosition.x,
        CONFIG.camera.initialPosition.y,
        CONFIG.camera.initialPosition.z
    );
    state.camera.lookAt(0, 0, 0);

    // Renderer
    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(container.clientWidth, container.clientHeight);
    state.renderer.setPixelRatio(window.devicePixelRatio);
    state.renderer.shadowMap.enabled = true;
    container.appendChild(state.renderer.domElement);

    // Lighting
    setupLighting();

    return { scene: state.scene, camera: state.camera, renderer: state.renderer };
}

function setupLighting() {
    const { ambient, main, fill, back } = CONFIG.lighting;
    
    // Ambient light
    const ambientLight = new THREE.AmbientLight(ambient.color, ambient.intensity);
    state.scene.add(ambientLight);

    // Main directional light
    const mainLight = new THREE.DirectionalLight(main.color, main.intensity);
    mainLight.position.set(...main.position);
    mainLight.castShadow = true;
    state.scene.add(mainLight);

    // Fill light
    const fillLight = new THREE.DirectionalLight(fill.color, fill.intensity);
    fillLight.position.set(...fill.position);
    state.scene.add(fillLight);

    // Back light
    const backLight = new THREE.DirectionalLight(back.color, back.intensity);
    backLight.position.set(...back.position);
    state.scene.add(backLight);
}

export function onWindowResize() {
    const container = document.getElementById('canvas-container');
    state.camera.aspect = container.clientWidth / container.clientHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(container.clientWidth, container.clientHeight);
}