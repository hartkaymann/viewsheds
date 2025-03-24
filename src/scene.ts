import { vec3 } from "gl-matrix";
import { Camera } from "./Camera";
import { QuadTree } from "./Optimization";
import { Bounds } from "./types/types";
import { Gizmo } from "./Gizmo";

export class Scene {

    camera: Camera

    points: Float32Array;
    colors: Float32Array;
    indices: Uint32Array;
    nodeToTriangles: Uint32Array;

    triangleCount: number;
    bounds: Bounds;
    tree: QuadTree;

    gizmo: Gizmo;

    constructor(camera: Camera) {
        this.camera = camera;

        this.points = new Float32Array();
        this.colors = new Float32Array();
        this.indices = new Uint32Array();
        this.nodeToTriangles = new Uint32Array();
        this.triangleCount = 0;
        this.bounds = {
            min: { x: 0, y: 0, z: 0 },
            max: { x: 0, y: 0, z: 0 }
        };
        this.tree = new QuadTree({ pos: vec3.create(), size: vec3.create() }, 0);

        this.gizmo = new Gizmo();
    }

    focusCameraOnPointCloud() {
        const bounds = this.bounds;

        const centerX = (bounds.min.x + bounds.max.x) / 2;
        const centerY = (bounds.min.y + bounds.max.y) / 2;
        const centerZ = (bounds.min.z + bounds.max.z) / 2;

        // Move the camera back along the Z-axis to fit the whole cloud in view
        const distance = Math.max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z) * 1.5;

        this.camera.setPosition(vec3.fromValues(centerX, centerY, centerZ + distance));
        this.camera.setTarget(vec3.fromValues(centerX, centerY, centerZ));
    }
}