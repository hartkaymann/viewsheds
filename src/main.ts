import { Scene } from "./Scene";
import { Camera } from "./Camera";
import { InputHandler } from "./InputHandler";
import { DeviceManager } from "./DeviceManager";
import { SceneLoader } from "./SceneLoader";
import { QuadTree } from "./Optimization";

import SceneWorker from './workers/SceneWorker.ts?worker';
import { Utils } from "./Utils";
import { UIController } from "./ui/UIController";
import { Controller } from "./Controller";
import { SceneSyncer } from "./SceneSyncer";
import { WorkspaceManager } from "./ui/WorkspaceManager";
import { ImportControls } from "./ui/ImportControls";

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

    const workspaceManager = new WorkspaceManager(controller);
    await workspaceManager.init();

    const uiController = new UIController(controller);
    await uiController.init();
    controller.ui = uiController;

    const importControls = new ImportControls();

    const sceneLoader = setupSceneLoader();

    importControls.onFileSelected((file) => {
        const reader = new FileReader();
        reader.onload = () => {
            sceneLoader.peekHeader(reader.result as ArrayBuffer, file.name);
        };
        reader.readAsArrayBuffer(file);
    });

    importControls.onConfirm(() => {
        controller.scene.clear();
        controller.reset();
        importControls.setProgress("Loading points", 0);
        sceneLoader.confirmLoad();
    });

    const treeDepth = 5; // Don't set above 8!

    function setupSceneLoader(): SceneLoader {
        const newLoader = new SceneLoader(SceneWorker);
        newLoader.setCallbacks({
            onHeader: (summary) => {
                importControls.showHeader(summary);
            },

            onProgress: (stage, value) => {
                importControls.setProgress(stage, value);
            },

            onPointsLoaded: ({ points, colors, classification, bounds }) => {
                controller.scene.points = points;
                controller.scene.colors = colors;
                controller.scene.classification = classification;
                controller.scene.bounds = bounds;

                controller.viewports.focusCameraOnPointCloud();
                controller.setPointData();

                const origin: [number, number, number] = [
                    (bounds.min.x + bounds.max.x) / 2,
                    (bounds.min.y + bounds.max.y) + 10,
                    (bounds.min.z + bounds.max.z) / 2,
                ];
                controller.updateRayOrigin(origin);
                uiController.updateOriginInputs(origin);
                uiController.handleUpdateRaySamples();

                importControls.complete();

                newLoader.startTreeBuild(points, bounds, treeDepth);
            },

            onTreeBuilt: (treeData) => {
                controller.scene.tree = QuadTree.reconstructProfiled(treeData, treeDepth);
                controller.setNodeData().then(() => {
                    uiController.setRunNodesButtonDisabled(false);
                });;
            },

            onTriangulationDone: ({ indices, triangleCount, treeData, nodeToTriangles }) => {
                controller.scene.indices = indices;
                controller.scene.triangleCount = triangleCount;
                controller.scene.tree = QuadTree.reconstructProfiled(treeData, treeDepth);
                controller.scene.nodeToTriangles = nodeToTriangles;

                controller.setNodeData(false);
                controller.setMeshData().then(() => {
                    uiController.setRunPointsButtonDisabled(false);
                    uiController.setRunPanoramaButtonDisabled(false);
                });

                console.log("Triangulation complete");
            },

            onError: (error) => {
                console.error("Scene load failed:", error);
                importControls.reset();
            }
        });

        return newLoader;
    }

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
}

window.addEventListener("DOMContentLoaded", () => {
    main().catch(console.error);
  });