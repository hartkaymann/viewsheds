import { vec3 } from "gl-matrix";
import { Camera } from "./Camera";
import { QuadTree } from "./Optimization";
import { Bounds } from "./types/types";
import { Gizmo } from "./Gizmo";

interface RaysConfig {
    origin: [number, number, number],
    theta: [number, number],
    phi: [number, number],
    samples: [number, number]
}

export class Scene {

    points: Float32Array = new Float32Array();
    colors: Float32Array = new Float32Array();
    indices: Uint32Array = new Uint32Array();
    classification: Uint32Array = new Uint32Array();
    nodeToTriangles: Uint32Array = new Uint32Array();

    triangleCount: number = 0;
    bounds: Bounds = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 }
    };
    tree: QuadTree = new QuadTree({ pos: vec3.create(), size: vec3.create() }, 0);

    rays: RaysConfig = {
        origin: [0, 0, 0],
        samples: [1, 1],
        theta: [0, 0],
        phi: [0, 0]
    };


    constructor() { }

    clear() {
        this.points = new Float32Array();
        this.colors = new Float32Array();
        this.indices = new Uint32Array();
        this.classification = new Uint32Array();
        this.nodeToTriangles = new Uint32Array();
        this.triangleCount = 0;
        this.bounds = {
            min: { x: 0, y: 0, z: 0 },
            max: { x: 0, y: 0, z: 0 }
        };
        this.tree = new QuadTree({ pos: vec3.create(), size: vec3.create() }, 0);
        this.rays = {
            origin: [0, 0, 0],
            samples: [1, 1],
            theta: [0, 0],
            phi: [0, 0]
        }
    }
}
