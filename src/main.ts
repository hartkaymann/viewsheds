import { Scene } from "./Scene";
import { Renderer } from "./Renderer";
import { Camera } from "./Camera";
import { InputHandler } from "./InputHandler";
import { DeviceManager } from "./DeviceManager";
import { SceneLoader } from "./SceneLoader";
import { QuadTree } from "./Optimization";

import SceneWorker from './workers/SceneWorker.ts?worker';

async function main() {
    const deviceManager = new DeviceManager();
    try {
        await deviceManager.init();
    } catch (e) {
        console.error("WebGPU init failed:", e);
        return; // TODO: Show fallback UI
    }

    const device = deviceManager.getDevice();
    const canvas: HTMLCanvasElement = <HTMLCanvasElement>document.getElementById("gfx-main");

    const camera: Camera = new Camera(
        [10, 10, 10],
        [0, 0, 0],
        [0, 1, 0],
        Math.PI / 4,
        canvas.width / canvas.height,
        0.1,
        10000
    );

    const scene: Scene = new Scene(camera);
    const renderer = new Renderer(canvas, scene, device);
    const inputHandler = new InputHandler(canvas, camera, scene);

    await renderer.init();
    renderer.startRendering();

    let sceneLoader = setupSceneLoader();
    
    // TODO: rather do the on change per setting than update values all at once!
    const treeDepth = 6; // Don't set above 8! 

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
                renderer.setNodeData();
    
                newLoader.startTriangulation(scene.points, treeData, treeDepth);
            },
    
            onTriangulationDone: ({ indices, triangleCount, nodeToTriangles }) => {
                scene.indices = indices;
                scene.triangleCount = triangleCount;
                scene.nodeToTriangles = nodeToTriangles;
    
                renderer.setMeshData();
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
            sceneLoader.worker.postMessage({ type: "shutdown" });
            sceneLoader = setupSceneLoader();
            
            reader.readAsArrayBuffer(file);
        }
    });

    document.getElementById("clear-cache")?.addEventListener("click", () => {
        const dbName = "PointCloudCache";
    
        const request = indexedDB.deleteDatabase(dbName);
    
        request.onsuccess = () => {
            console.log(`IndexedDB "${dbName}" deleted successfully.`);
        };
    
        request.onerror = () => {
            console.error(`Failed to delete IndexedDB "${dbName}".`, request.error);
        };
    
        request.onblocked = () => {
            console.warn(`Delete blocked. Please close all other tabs using this database.`);
        };
    });

    const url = "/model/80049_1525964_M-34-63-B-b-1-4-4-3.laz"
    sceneLoader.worker.postMessage({ type: "load-url", url });
}

    

main().catch(console.error);