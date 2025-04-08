import { Scene } from "./Scene";
import { Renderer } from "./Renderer";
import { Camera } from "./Camera";
import { InputHandler } from "./InputHandler";
import { DeviceManager } from "./DeviceManager";
import { SceneLoader } from "./SceneLoader";
import { QuadTree } from "./Optimization";

import SceneWorker from './workers/SceneWorker.ts?worker';
import { Utils } from "./Utils";
import { UIController } from "./ui/UIController ";

declare global {
    interface Window {
        _consoleError?: typeof console.error;
        _consoleWarn?: typeof console.warn;
    }
}

async function main() {
    // Setup up toasts
    window._consoleError = window.console.error;
    window._consoleWarn = window.console.warn;

    console.error = (...args) => {
        Utils.showToast(args.join(' '), 'error');
        window._consoleError?.(...args);
    };

    console.warn = (...args) => {
        Utils.showToast(args.join(' '), 'warn');
        window._consoleWarn?.(...args);
    };

    // Create device
    const deviceManager = new DeviceManager();
    try {
        await deviceManager.init();
    } catch (e) {
        console.error("WebGPU init failed:", e);

        const fallback = document.getElementById("fallback-message");
        const canvas = document.getElementById("gfx-main");
        if (fallback && canvas) {
            fallback.style.display = "block";
            canvas.style.display = "none";
        }

        return; // TODO: Show fallback UI
    }

    const device = deviceManager.getDevice();

    const canvas: HTMLCanvasElement = <HTMLCanvasElement>document.getElementById("gfx-main");
    const wrapper = document.getElementById('canvas-wrapper')!;

    const camera: Camera = new Camera(
        [10, 10, 10],
        [0, 0, 0],
        [0, 1, 0],
        Math.PI / 4,
        canvas.width / canvas.height,
        0.1,
        10000
    );

    const devicePixelRatio = window.devicePixelRatio || 1;
    function updateCanvasSize() {
        const width = Math.floor(wrapper.clientWidth * devicePixelRatio);
        const height = Math.floor(wrapper.clientHeight * devicePixelRatio);

        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
            renderer.resize(width, height);
        }

        camera.aspect = width / height;
        camera.setProjection();
    }

    const resizeObserver = new ResizeObserver(updateCanvasSize);
    resizeObserver.observe(wrapper);


    const scene: Scene = new Scene(camera);
    const inputHandler = new InputHandler(canvas, camera, scene);

    const renderer = new Renderer(canvas, scene, device);
    const uiController = new UIController(renderer);
    
    renderer.ui = uiController;
    await renderer.init();
    uiController.init();
    
    renderer.startRendering();

    let sceneLoader = setupSceneLoader();

    const treeDepth = 8; // Don't set above 8! 

    function setupSceneLoader(): SceneLoader {
        const newLoader = new SceneLoader(SceneWorker);
        newLoader.setCallbacks({
            onPointsLoaded: ({ points, colors, classification, bounds }) => {
                scene.points = points;
                scene.colors = colors;
                scene.classification = classification;
                scene.bounds = bounds;

                scene.focusCameraOnPointCloud();
                renderer.setPointData();

                newLoader.startTreeBuild(points, bounds, treeDepth);
            },

            onTreeBuilt: (treeData) => {
                scene.tree = QuadTree.reconstruct(treeData, treeDepth);
                renderer.setNodeData().then(() => {
                    const runNodesButton = document.getElementById("runNodes") as HTMLButtonElement;
                    if (runNodesButton) {
                        runNodesButton.disabled = false;
                    };
                });;
            },

            onTriangulationDone: ({ indices, triangleCount, treeData, nodeToTriangles }) => {
                scene.indices = indices;
                scene.triangleCount = triangleCount;
                scene.tree = QuadTree.reconstruct(treeData, treeDepth);
                scene.nodeToTriangles = nodeToTriangles;

                renderer.setNodeData(false);
                renderer.setMeshData().then(() => {
                    const runPointsButton = document.getElementById("runPoints") as HTMLButtonElement;
                    if (runPointsButton) {
                        runPointsButton.disabled = false;
                    };

                    const runPanoramaButton = document.getElementById("runPanorama") as HTMLButtonElement;
                    if (runPanoramaButton) {
                        runPanoramaButton.disabled = false;
                    };
                });

                console.log("Triangulation complete");
            }
        });

        return newLoader;
    }

    const input = document.getElementById("file-input") as HTMLInputElement;
    input.addEventListener("change", (event) => {
        const file = input.files?.[0];
        if (file) {
            const reader = new FileReader();

            reader.onload = () => {
                const arrayBuffer = reader.result as ArrayBuffer;

                sceneLoader.worker.postMessage({
                    type: "load-arraybuffer",
                    name: file.name,
                    buffer: arrayBuffer
                }, [arrayBuffer]);
            };

            scene.clear();
            renderer.reset();

            sceneLoader.shutdown();
            sceneLoader = setupSceneLoader();

            reader.readAsArrayBuffer(file);
        }
    });

    document.getElementById("clear-cache")?.addEventListener("click", () => {
        const dbName = "PointCloudCache";

        const request = indexedDB.deleteDatabase(dbName);

        request.onsuccess = () => {
            let msg = `IndexedDB "${dbName}" deleted successfully.`;
            console.log(msg);
            Utils.showToast(msg, "info");
        };

        request.onerror = () => {
            console.error(`Failed to delete IndexedDB "${dbName}".`, request.error);
        };

        request.onblocked = () => {
            console.warn(`Delete blocked. Please close all other tabs using this database.`);
        };

    });

    const response = await fetch(`${import.meta.env.BASE_URL}model/files.json`);
    const lazFiles = await response.json();
    if (lazFiles.length === 0) {
        console.warn('No .laz files found.');
        return;
    }
    const url = new URL(`${import.meta.env.BASE_URL}model/${lazFiles[0]}`, location.origin).toString();;
    console.log('First LAZ file URL:', url);
    sceneLoader.worker.postMessage({ type: "load-url", url });
}

main().catch(console.error);