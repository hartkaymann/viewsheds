import { vec2, vec3 } from "gl-matrix";
import { Camera } from "./camera";

namespace icosahedron {
    const X: number = 0.525731112119133606;
    const Z: number = 0.850650808352039932;
    const N: number = 0.0;

    export const vertices: Float32Array = new Float32Array([
        -X, N, Z,  
        X, N, Z, 
        -X, N, -Z,  
        X, N, -Z,

        N, Z, X,  
        N, Z, -X,  
        N, -Z, X,  
        N, -Z, -X,

        Z, X, N, 
        -Z, X, N,  
        Z, -X, N, 
        -Z, -X, N
    ]);

    export const triangles: Uint32Array = new Uint32Array([
        0, 4, 1,  
        0, 9, 4,  
        9, 5, 4,  
        4, 5, 8,  
        4, 8, 1,
        8, 10, 1,  
        8, 3, 10,  
        5, 3, 8,  
        5, 2, 3,  
        2, 7, 3,
        7, 10, 3,  
        7, 6, 10,  
        7, 11, 6,  
        11, 0, 6,  
        0, 1, 6,
        6, 1, 10,  
        9, 0, 11,  
        9, 11, 2,  
        9, 2, 5,  
        7, 2, 11
    ]);
}


export class Scene {

    camera: Camera

    points: Float32Array;
    indices: Uint32Array;
    triangleCount: number;

    constructor(camera: Camera) {
        this.camera = camera;
        this.points = icosahedron.vertices;
        this.indices = icosahedron.triangles;
        this.triangleCount = icosahedron.triangles.length / 3;

        // this.init();
    }

    init() {
        this.loadPLYFromURL("/model/galleon.ply");
    }

    async loadPLYFromURL(url: string): Promise<void> {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to load file: ${response.statusText}`);

            const data = await response.text();
            this.points = new Float32Array(this.parsePLY(data));
            console.log("Loaded points from URL:", this.points);
        } catch (error) {
            console.error("Error loading PLY from URL:", error);
        }
    }

    private parsePLY(data: string): number[] {
        const lines = data.split("\n");

        const vertexLine = lines.find(line => line.startsWith("element vertex"));
        if (!vertexLine) throw new Error("Invalid PLY file: No vertex count found.");

        const vertexCount = parseInt(vertexLine.split(" ")[2], 10); // Extract vertex count
        if (isNaN(vertexCount)) throw new Error("Invalid vertex count in PLY file.");

        const headerIndex = lines.findIndex(line => line.trim() === "end_header");
        if (headerIndex === -1) throw new Error("Invalid PLY file");

        return lines
            .slice(headerIndex + 1, headerIndex + 1 + vertexCount)
            .map(line => line.trim().split(/\s+/).map(Number))
            .filter(coords => coords.length >= 3)
            .flat();
    }
}