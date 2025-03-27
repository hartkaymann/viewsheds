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

    const sceneLoader = new SceneLoader(SceneWorker);
    const treeDepth = 5; // Don't set above 8! 
    sceneLoader.setCallbacks({
        onPointsLoaded: ({ points, colors, bounds }) => {
            scene.points = points;
            scene.colors = colors;
            scene.bounds = bounds;
            
            scene.focusCameraOnPointCloud();
            renderer.setPointData();

            sceneLoader.startTreeBuild(points, bounds, treeDepth);
        },

        onTreeBuilt: (treeData) => {
            scene.tree = QuadTree.reconstruct(treeData, treeDepth);
            renderer.setNodeData();

            sceneLoader.startTriangulation(scene.points, treeData, treeDepth);
        },

        onTriangulationDone: ({ indices, triangleCount, nodeToTriangles }) => {
            scene.indices = indices;
            scene.triangleCount = triangleCount;
            scene.nodeToTriangles = nodeToTriangles;

            renderer.setMeshData();
            console.log("Triangulation complete");
        }
    });

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
            
            reader.readAsArrayBuffer(file);
        }
    });
    const url = "/model/80049_1525964_M-34-63-B-b-1-4-4-3.laz"
    sceneLoader.worker.postMessage({ type: "load-url", url });
}
    

main().catch(console.error);