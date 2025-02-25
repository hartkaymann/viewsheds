import { vec3, mat4 } from "gl-matrix"

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

    constructor(theta: number, phi: number, radius: number, up: number[], center: number[], fov: number, aspect: number, near: number, far: number) {
        this.theta = theta;
        this.phi = phi;
        this.radius = radius;

        this.worldUp = new Float32Array(up)
        this.center = new Float32Array(center);
        this.target = new Float32Array(center);

        this.position = [0, 0, 0];
        this.front = [0, 0, 0];
        this.right = [0, 0, 0];
        this.up = [0, 0, 0];
        
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
        let temp = vec3.create();;

        // Update front
        let x = Math.sin(this.theta) * Math.sin(this.phi);
        let y = Math.cos(this.phi);
        let z = Math.cos(this.theta) * Math.sin(this.phi);
        vec3.normalize(this.front, [x, y, z]);

        // Update position
        vec3.scale(temp, [x, y, z], this.radius);
        vec3.add(this.position, this.target, temp)
        
        // Update right
        vec3.cross(temp, this.front, this.worldUp);
        vec3.normalize(this.right, temp);

        // Update up
        vec3.cross(temp, this.right, this.front);
        vec3.normalize(this.up, temp);
    }    

    recalculate_matrices() {
        mat4.lookAt(this.viewMatrix, this.position, this.target, this.up);
        mat4.perspective(this.projectionMatrix, this.fov, this.aspect, this.near, this.far);
    }

    orbit(deltaTheta: number, deltaPhi: number) {
        this.theta += deltaTheta;
        this.phi += deltaPhi;

        // Clamp phi so camera doesn't flipping over
        const epsilon = 0.001;
        this.phi = Math.max(epsilon, Math.min(Math.PI - epsilon, this.phi));

        this.update();
    }

    zoom(deltaZoom: number) {
        this.radius = Math.max(0.5, Math.min(1000, this.radius + deltaZoom)); 
        this.update();
    }
}