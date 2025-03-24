import { Scene } from "./Scene";
import { Renderer } from "./Renderer";
import { Camera } from "./Camera";
import { InputHandler } from "./InputHandler";
import { DeviceManager } from "./DeviceManager";
import { SceneLoader } from "./SceneLoader";

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

    const loader = new SceneLoader();
    const url = "./model/80049_1525964_M-34-63-B-b-1-4-4-3.laz";
    loader.fetchAndProcessLASorLAZ(url).then(async ({ points, colors, bounds }) => {
        scene.points = points;
        scene.colors = colors;
        scene.bounds = bounds;

        renderer.setPointData();

        // Step 2: Generate quadtree
        loader.createQuadtree(points, bounds).then(tree => {
            scene.tree = tree;
            renderer.setNodeData();

            // // Step 3: Triangulate
            // loader.performTriangulation(points, tree).then(({ indices, triangleCount, nodeToTriangles }) => {
            //     scene.indices = indices;
            //     scene.triangleCount = triangleCount;
            //     scene.nodeToTriangles = nodeToTriangles;

            //     renderer.setMeshData();
            //     console.log("Triangulation complete");
            // });
        });
    });
}

    
main().catch(console.error);