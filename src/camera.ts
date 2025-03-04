import { vec3, mat4 } from "gl-matrix";

export class Camera {
    position: vec3;
    front: vec3;
    right: vec3;
    up: vec3;
    center: vec3;
    worldUp: vec3;

    fov: number;
    aspect: number;
    near: number;
    far: number;
    radius: number;
    theta: number;
    phi: number;

    target: vec3;

    projectionMatrix: mat4;
    viewMatrix: mat4;

    constructor(theta: number, phi: number, radius: number, up: vec3, center: vec3, fov: number, aspect: number, near: number, far: number) {
        this.theta = theta;
        this.phi = phi;
        this.radius = radius;

        this.worldUp = vec3.clone(up);
        this.center = vec3.clone(center);
        this.target = vec3.clone(center);

        this.position = vec3.create();
        this.front = vec3.create();
        this.right = vec3.create();
        this.up = vec3.create();

        this.fov = fov;
        this.aspect = aspect;
        this.near = near;
        this.far = far;

        this.projectionMatrix = mat4.create();
        this.viewMatrix = mat4.create();

        this.update();
    }

    update() {
        this.recalculate_vectors();
        this.recalculate_matrices();
    }

    recalculate_vectors() {
        const temp = vec3.create();

        // Update front
        const x = Math.sin(this.theta) * Math.sin(this.phi);
        const y = Math.cos(this.phi);
        const z = Math.cos(this.theta) * Math.sin(this.phi);
        vec3.set(this.front, x, y, z); // TODO: Update fix, set radius when new target/positoin? Should cover it here?
        vec3.normalize(this.front, this.front);

        // Update position
        vec3.scale(temp, this.front, this.radius);
        vec3.sub(this.position, this.target, temp);

        // Update right
        vec3.cross(temp, this.worldUp, this.front);
        vec3.normalize(this.right, temp);

        // Update up
        vec3.cross(temp, this.front, this.right);
        vec3.normalize(this.up, temp);
    }

    recalculate_matrices() {
        mat4.lookAt(this.viewMatrix, this.position, this.target, this.up);
        mat4.perspective(this.projectionMatrix, this.fov, this.aspect, this.near, this.far);
    }

    setPosition(position: vec3) {
        vec3.copy(this.position, position);
        this.radius = vec3.distance(this.position, this.target);
        this.update();
    }

    setTarget(target: vec3) {
        vec3.copy(this.target, target);
        this.radius = vec3.distance(this.position, this.target);
        this.update();
    }

    orbit(deltaTheta: number, deltaPhi: number) {
        this.theta += deltaTheta;
        this.phi += deltaPhi;

        // Clamp phi so camera doesn't flip over
        const epsilon = 0.001;
        this.phi = Math.max(epsilon, Math.min(Math.PI - epsilon, this.phi));
        this.update();
    }

    zoom(deltaZoom: number) {
        this.radius = Math.max(0.5, this.radius + deltaZoom);
        this.update();
    }
}