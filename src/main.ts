import { Scene } from "./scene";
import { Renderer } from "./renderer";
import { Camera } from "./camera";
import { InputHandler } from "./input-handler";

async function main() {

    const canvas: HTMLCanvasElement = <HTMLCanvasElement>document.getElementById("gfx-main");

    const camera: Camera = new Camera(
        0,
        Math.PI / 2,
        10,
        [0, 1, 0],
        [0, 0, 0],
        45,
        canvas.width / canvas.height,
        0.1,
        1000
    );

    const scene: Scene = new Scene(camera);
    const inputHandler = new InputHandler(canvas, camera, scene);
    const renderer = new Renderer(canvas, scene);

    console.log("Initializing scene...");
    await scene.init();

    renderer.init();
}

main().catch(console.error);