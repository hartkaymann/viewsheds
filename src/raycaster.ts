import { mat4, vec3 } from "gl-matrix";
import { Camera } from "./camera";

export type Ray = {
    origin: vec3,
    direction: vec3
}

export class Raycaster {
    canvas: HTMLCanvasElement;
    camera: Camera;

    prevRay: Ray | null;
    deltaRay: Ray;


    constructor(canvas: HTMLCanvasElement, camera: Camera) {
        this.canvas = canvas;
        this.camera = camera;

        this.prevRay = null;
        this.deltaRay = {origin: [0, 0, 0], direction: [0, 0, 0]};
    }

    getMouseNDC(event: MouseEvent) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width * 2 - 1 // [-1, 1] range
        const y = -((event.clientY - rect.top) / rect.height * 2 - 1) // [-1, 1] range, inverted

        return { x, y }
    }

    createRayFromMouse(event: MouseEvent): Ray {
        const ndc = this.getMouseNDC(event);
        const nearPoint: vec3 = [ndc.x, ndc.y, -1.0];
        const farPoint: vec3 = [ndc.x, ndc.y, 1.0];

        const invProjView = mat4.create();
        mat4.mul(invProjView, this.camera.projectionMatrix, this.camera.viewMatrix);
        mat4.invert(invProjView, invProjView);

        // Unproject points NDC to world
        const nearWorld = vec3.transformMat4(vec3.create(), nearPoint, invProjView);
        const farWorld = vec3.transformMat4(vec3.create(), farPoint, invProjView);

        const rayDirection = vec3.sub(vec3.create(), farWorld, nearWorld);
        vec3.normalize(rayDirection, rayDirection);

        return { origin: nearWorld, direction: rayDirection };
    }

    updateRay(event: MouseEvent) {
        const currentRay = this.createRayFromMouse(event);

        if (this.prevRay) {
            const deltaOrigin = vec3.sub(vec3.create(), currentRay.origin, this.prevRay.origin);
            const deltaDirection = vec3.sub(vec3.create(), currentRay.direction, this.prevRay.direction);

            this.deltaRay = { origin: deltaOrigin, direction: deltaDirection };
        }

        this.prevRay = currentRay;
        return currentRay;
    }

    intersectSphere(origin: vec3, direction: vec3, position: vec3, radius: number): boolean {
        const oc = vec3.sub(vec3.create(), origin, position);
        const a = vec3.dot(direction, direction);
        const b = 2.0 * vec3.dot(oc, direction);
        const c = vec3.dot(oc, oc) - radius * radius;
        const discriminant = b * b - 4 * a * c;

        return discriminant > 0;
    }
}