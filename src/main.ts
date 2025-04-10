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
import { Controller } from "./Controller";
import { SceneSyncer } from "./SceneSyncer";

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

    const controller = new Controller(device)
    await controller.init();
    controller.start();

    const uiController = new UIController(controller);
    await uiController.init();

    let sceneLoader = setupSceneLoader();

    const treeDepth = 8; // Don't set above 8! 

    function setupSceneLoader(): SceneLoader {
        const newLoader = new SceneLoader(SceneWorker);
        newLoader.setCallbacks({
            onPointsLoaded: ({ points, colors, classification, bounds }) => {
                controller.scene.points = points;
                controller.scene.colors = colors;
                controller.scene.classification = classification;
                controller.scene.bounds = bounds;

                controller.viewports.focusCameraOnPointCloud();
                controller.setPointData();

                newLoader.startTreeBuild(points, bounds, treeDepth);
            },

            onTreeBuilt: (treeData) => {
                controller.scene.tree = QuadTree.reconstruct(treeData, treeDepth);
                controller.setNodeData().then(() => {
                    uiController.setRunNodesButtonDisabled(false);
                });;
            },

            onTriangulationDone: ({ indices, triangleCount, treeData, nodeToTriangles }) => {
                controller.scene.indices = indices;
                controller.scene.triangleCount = triangleCount;
                controller.scene.tree = QuadTree.reconstruct(treeData, treeDepth);
                controller.scene.nodeToTriangles = nodeToTriangles;

                controller.setNodeData(false);
                controller.setMeshData().then(() => {
                    uiController.setRunPointsButtonDisabled(false);
                    uiController.setRunPanoramaButtonDisabled(false);
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

            controller.scene.clear();
            controller.reset();

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

window.addEventListener("DOMContentLoaded", () => {
    main().catch(console.error);
  });