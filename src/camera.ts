import { vec3, mat4, quat } from "gl-matrix";

export class Camera {
    position: vec3;
    target: vec3;
    right: vec3;
    up: vec3;
    worldUp: vec3;

    orientation: quat;

    rotationSpeed: number;
    panSpeed: number;
    zoomSpeed: number;

    fov: number;
    aspect: number;
    near: number;
    far: number;

    projectionMatrix: mat4;
    viewMatrix: mat4;

    constructor(position: vec3, target: vec3, up: vec3, fov: number, aspect: number, near: number, far: number) {
        this.position = vec3.clone(position);
        this.target = vec3.clone(target);
        this.up = vec3.clone(up);

        this.fov = fov;
        this.aspect = aspect;
        this.near = near;
        this.far = far;

        this.projectionMatrix = mat4.create();
        this.viewMatrix = mat4.create();

        this.orientation = quat.create();
        let direction = vec3.create();
        vec3.sub(direction, this.target, this.position);
        vec3.normalize(direction, direction);
        quat.rotationTo(this.orientation, vec3.fromValues(0, 0, -1), direction);

        this.rotationSpeed = 0.01;
        this.panSpeed = 0.2;
        this.zoomSpeed = 0.3;

        this.updateView();
        this.setProjection();
    }

    updateView() {
        mat4.lookAt(this.viewMatrix, this.position, this.target, this.up);
    }

    setProjection() {
        mat4.perspective(this.projectionMatrix, this.fov, this.aspect, this.near, this.far);
    }

    setPosition(position: vec3) {
        vec3.copy(this.position, position);
        this.updateView();
    }

    setTarget(target: vec3) {
        vec3.copy(this.target, target);
        this.updateView();
    }

    rotate(deltaX: number, deltaY: number) {
        const pitchAngle = deltaX * this.rotationSpeed;
        const yawAngle = -deltaY * this.rotationSpeed;

        const yawQuat = quat.create();
        quat.setAxisAngle(yawQuat, vec3.fromValues(0, 1, 0), pitchAngle);

        let forward = vec3.create();
        vec3.sub(forward, this.target, this.position);
        vec3.normalize(forward, forward);

        let right = vec3.create();
        if (Math.abs(vec3.dot(this.up, forward)) > 0.9999) {
            //Gimal lock situation
            vec3.cross(right, vec3.fromValues(1, 0, 0), this.up);
        }
        else {
            vec3.cross(right, this.up, forward);
        }
        vec3.normalize(right, right);

        const pitchQuat = quat.create();
        quat.setAxisAngle(pitchQuat, right, yawAngle);

        const rotation = quat.create();
        quat.multiply(rotation, rotation, pitchQuat);
        quat.multiply(rotation, yawQuat, rotation);
        quat.normalize(this.orientation, rotation);

        let initialOffset = vec3.create();
        vec3.sub(initialOffset, this.position, this.target);

        let rotatedOffset = vec3.create();
        vec3.transformQuat(rotatedOffset, initialOffset, this.orientation);

        vec3.add(this.position, this.target, rotatedOffset);

        this.updateView();
    }

    pan(deltaX: number, deltaY: number) {
        const panX = deltaX * this.panSpeed;
        const panY = deltaY * this.panSpeed;

        let direction = vec3.create();
        vec3.sub(direction, this.target, this.position);
        vec3.normalize(direction, direction);

        let right = vec3.create();
        vec3.cross(right, this.up, direction);
        vec3.normalize(right, right);

        let up = vec3.create();
        vec3.cross(up, right, direction);
        vec3.normalize(up, up);

        let panOffset = vec3.create();
        vec3.scaleAndAdd(panOffset, panOffset, right, panX);
        vec3.scaleAndAdd(panOffset, panOffset, up, panY);

        vec3.add(this.position, this.position, panOffset);
        vec3.add(this.target, this.target, panOffset);

        this.updateView();
    }

    zoom(deltaZoom: number) {
        let direction = vec3.create();
        vec3.sub(direction, this.target, this.position);
        vec3.normalize(direction, direction);

        vec3.add(this.position, this.position, vec3.scale(vec3.create(), direction, deltaZoom * this.zoomSpeed));
        this.updateView();
    }
}