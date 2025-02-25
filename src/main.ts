import { Scene } from "./scene";
import { Renderer } from "./renderer";
import { Camera } from "./camera";
import { event } from "jquery";
import { InputHandler } from "./input-handler";

const canvas: HTMLCanvasElement = <HTMLCanvasElement>document.getElementById("gfx-main");

const camera: Camera = new Camera(
    0,
    Math.PI / 2,
    5,
    [0, 1, 0],
    [0, 0, 0],
    45,
    canvas.width / canvas.height,
    0.1,
    100);
const scene: Scene = new Scene(camera);

const inputHandler = new InputHandler(canvas, camera, scene);

const renderer = new Renderer(canvas, scene);

renderer.init();

