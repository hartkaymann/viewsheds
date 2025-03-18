import { vec3 } from "gl-matrix";

export interface Bounds {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
}

export interface AABB {
    pos: vec3,
    size: vec3
}