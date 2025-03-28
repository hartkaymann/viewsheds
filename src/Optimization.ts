import { vec2, vec3 } from "gl-matrix";
import { AABB, Bounds } from "./types/types";

class QuadTreeNode {
    children: QuadTreeNode[] | null;
    bounds: AABB;
    index: number;

    startPointIndex: number;
    pointCount: number;

    startTriangleIndex: number;
    triangleCount: number;

    constructor(bounds: AABB, depth: number) {
        this.children = null;
        this.bounds = bounds;
        this.startPointIndex = -1;
        this.pointCount = 0;
        this.startTriangleIndex = -1;
        this.triangleCount = 0;
        this.index = -1;

        if (depth > 0) {
            const halfSize = vec3.fromValues(bounds.size[0] / 2, 0, bounds.size[2] / 2);

            this.children = [
                new QuadTreeNode({
                    pos: vec3.clone(bounds.pos),
                    size: vec3.clone(halfSize)
                }, depth - 1),

                new QuadTreeNode({
                    pos: vec3.fromValues(bounds.pos[0] + halfSize[0], bounds.pos[1], bounds.pos[2]),
                    size: vec3.clone(halfSize)
                }, depth - 1),

                new QuadTreeNode({
                    pos: vec3.fromValues(bounds.pos[0], bounds.pos[1], bounds.pos[2] + halfSize[2]),
                    size: vec3.clone(halfSize)
                }, depth - 1),

                new QuadTreeNode({
                    pos: vec3.fromValues(bounds.pos[0] + halfSize[0], bounds.pos[1], bounds.pos[2] + halfSize[2]),
                    size: vec3.clone(halfSize)
                }, depth - 1),
            ]
        }
    }

    containsPoint(px: number, pz: number): boolean {
        const EPSILON = Math.min(this.bounds.size[0], this.bounds.size[2]) * 0.001;
        return (
            px >= this.bounds.pos[0] - EPSILON &&
            px <= this.bounds.pos[0] + this.bounds.size[0] + EPSILON &&
            pz >= this.bounds.pos[2] - EPSILON &&
            pz <= this.bounds.pos[2] + this.bounds.size[2] + EPSILON
        );
    }

    // Recursive depth-first traversal
    traverse(callback: (node: QuadTreeNode) => void) {
        callback(this);

        if (this.children) {
            for (const child of this.children) {
                child.traverse(callback);
            }
        }
    }

    assignIndicesBreadthFirst(): void {
        let index = 0;
        const queue: QuadTreeNode[] = [this];

        while (queue.length > 0) {
            const node = queue.shift()!;
            node.index = index++;

            if (node.children) {
                queue.push(...node.children);
            }
        }
    }

    assignPoints(sortedPoints: Float32Array, startIndex: number, endIndex: number): void {
        if (this.children === null) { // Leaf
            this.startPointIndex = startIndex;
            this.pointCount = endIndex - startIndex;

            if (this.pointCount == 0) {
                console.warn("No points in leaf:", this.bounds);
            }

            let yMin = Infinity;
            let yMax = -Infinity;

            for (let i = startIndex; i < endIndex; i++) {
                const y = sortedPoints[i * 4 + 1];
                yMin = Math.min(yMin, y);
                yMax = Math.max(yMax, y);
            }

            this.bounds.pos[1] = yMin;
            this.bounds.size[1] = yMax - yMin;

            return;
        }

        this.startPointIndex = startIndex;
        this.pointCount = 0;

        this.bounds.pos[1] = Infinity;
        this.bounds.size[1] = 0;

        let currentStart = startIndex;
        for (const child of this.children) {
            let newEnd = currentStart;

            while (newEnd < endIndex) {
                const offset = newEnd * 4;
                let px = sortedPoints[offset], pz = sortedPoints[offset + 2];

                if (child.containsPoint(px, pz)) {
                    newEnd++;
                } else {
                    break;
                }
            }

            if (newEnd > currentStart) {
                child.assignPoints(sortedPoints, currentStart, newEnd);

                this.pointCount += child.pointCount;

                this.bounds.pos[1] = Math.min(this.bounds.pos[1], child.bounds.pos[1]);
                this.bounds.size[1] = Math.max(this.bounds.size[1], (child.bounds.pos[1] + child.bounds.size[1]) - this.bounds.pos[1]);

                currentStart = newEnd;
            }
            else {
                 console.warn("No points assigned to child node:", child.bounds);
            }
        }
    }

    assignTriangles(
        triangles: Uint32Array,
        points: Float32Array,
        globalTriangleIndexBuffer: number[],
        relevantTriangles: Uint32Array | null = null
    ) {
        let trianglesToProcess = relevantTriangles ?? triangles;

        if (this.children === null) { // Leaf node
            this.startTriangleIndex = globalTriangleIndexBuffer.length;
            this.triangleCount = 0;

            for (let i = 0; i < trianglesToProcess.length; i += 3) {
                let v0 = trianglesToProcess[i], v1 = trianglesToProcess[i + 1], v2 = trianglesToProcess[i + 2];

                let base0 = v0 * 4, base1 = v1 * 4, base2 = v2 * 4;
                let p0x = points[base0], p0z = points[base0 + 2];
                let p1x = points[base1], p1z = points[base1 + 2];
                let p2x = points[base2], p2z = points[base2 + 2];

                if (this.containsPoint(p0x, p0z) || this.containsPoint(p1x, p1z) || this.containsPoint(p2x, p2z)) {
                    globalTriangleIndexBuffer.push(i / 3);
                    this.triangleCount++;
                }
            }
            return globalTriangleIndexBuffer;
        }

        let filteredTriangles: number[] = [];
        for (let i = 0; i < trianglesToProcess.length; i += 3) {
            let v0 = trianglesToProcess[i], v1 = trianglesToProcess[i + 1], v2 = trianglesToProcess[i + 2];

            let base0 = v0 * 4, base1 = v1 * 4, base2 = v2 * 4;
            let p0x = points[base0], p0z = points[base0 + 2];
            let p1x = points[base1], p1z = points[base1 + 2];
            let p2x = points[base2], p2z = points[base2 + 2];

            if (this.containsPoint(p0x, p0z) || this.containsPoint(p1x, p1z) || this.containsPoint(p2x, p2z)) {
                filteredTriangles.push(v0, v1, v2);
            }
        }

        let filteredTriangleArray = filteredTriangles.length > 0 ? new Uint32Array(filteredTriangles) : null;
        for (const child of this.children) {
            child.assignTriangles(filteredTriangleArray, points, globalTriangleIndexBuffer, filteredTriangleArray);
        }
    }
}

export class QuadTree {
    depth: number;
    root: QuadTreeNode;
    flat: ArrayBuffer | null = null;

    static readonly BYTES_PER_NODE = 48; // 12 floats

    constructor(bounds: AABB, depth: number) {
        this.depth = depth;
        this.root = new QuadTreeNode(bounds, depth);
    }

    assignPoints(sortedPoints: Float32Array): void {
        const pointSize = 4; // (x, y, z, w)
        this.root.assignPoints(sortedPoints, 0, sortedPoints.length / pointSize);
    }

    assignIndices() {
        this.root.assignIndicesBreadthFirst();
    }

    assignTriangles(triangles: Uint32Array, points: Float32Array, globalTriangleIndexBuffer: number[]): void {
        this.root.assignTriangles(triangles, points, globalTriangleIndexBuffer);
    }

    flatten(): ArrayBuffer {
        if (this.flat)
            return this.flat;

        const nodeList: QuadTreeNode[] = [];
        const queue: QuadTreeNode[] = [this.root];

        while (queue.length > 0) {
            const node = queue.shift()!;
            nodeList.push(node);

            if (node.children) {
                queue.push(...node.children);
            }
        }

        const buffer = new ArrayBuffer(nodeList.length * QuadTree.BYTES_PER_NODE);
        const floatView = new Float32Array(buffer); // Stores float values
        const intView = new Uint32Array(buffer); // Access integer part correctly

        nodeList.forEach((node, i) => {
            const offset = i * 12;
            floatView.set(node.bounds.pos, offset);
            intView[offset + 3] = node.children ? 4 : 0;
            floatView.set(node.bounds.size, offset + 4);
            intView[offset + 7] = node.startPointIndex;
            intView[offset + 8] = node.pointCount;
            intView[offset + 9] = node.startTriangleIndex;
            intView[offset + 10] = node.triangleCount;
        });

        this.flat = buffer;
        return buffer;
    }

    mapPointsToNodes = (): Uint32Array => {
        const pointToNodeBuffer = new Uint32Array(this.root.pointCount); // One index per point

        this.root.traverse(node => {
            if (node.children === null) { // Only assign leaf nodes
                for (let i = 0; i < node.pointCount; i++) {
                    pointToNodeBuffer[node.startPointIndex + i] = node.index; // Assign node ID
                }
            }
        });

        return pointToNodeBuffer;
    };

    static reconstruct(buffer: ArrayBuffer, depth: number): QuadTree {
        const floatView = new Float32Array(buffer);
        const intView = new Uint32Array(buffer);

        // Extract bounds from node 0

        const offset = 0;
        const pos = vec3.fromValues(
            floatView[offset],
            floatView[offset + 1],
            floatView[offset + 2]
        );
        const size = vec3.fromValues(
            floatView[offset + 4],
            floatView[offset + 5],
            floatView[offset + 6]
        );

        const bounds = { pos, size };
        const tree = new QuadTree(bounds, depth);
        tree.assignIndices();

        tree.root.traverse(node => {
            const i = node.index;
            const off = i * 12;

            node.bounds.pos[0] = floatView[off];
            node.bounds.pos[1] = floatView[off + 1];
            node.bounds.pos[2] = floatView[off + 2];

            const childCount = intView[off + 3];
            if (childCount === 0) {
                node.children = null;
            }

            node.bounds.size[0] = floatView[off + 4];
            node.bounds.size[1] = floatView[off + 5];
            node.bounds.size[2] = floatView[off + 6];

            node.startPointIndex = intView[off + 7];
            node.pointCount = intView[off + 8];
            node.startTriangleIndex = intView[off + 9];
            node.triangleCount = intView[off + 10];
        });

        return tree;

    }

    static noMaxNodesHit(depth: number): number {
        return 2 ** (depth + 1)
    }

    static totalNodes(depth: number): number {
        return (4 ** (depth + 1) - 1) / 3;
    }

    static leafNodes(depth: number): number {
        return 4 ** depth;
    }
}

export class MortonSorter {

    constructor() { }

    sort(
        points: Float32Array,
        bounds: Bounds): { sortedPoints: Float32Array, sortedIndices: Uint32Array } {

        const pointSize = 4; // Each point has 4 components (x, y, z, w)
        const numPoints = points.length / pointSize;

        // Compute Morton code for each point
        const mortonCodes = new Uint32Array(numPoints);
        const indices = new Uint32Array(numPoints);

        for (let i = 0; i < numPoints; i++) {
            const offset = i * pointSize;
            mortonCodes[i] = this.computeMortonCodeXZ(points[offset], points[offset + 2], bounds);
            indices[i] = i;
        }

        // Sort by Morton code
        indices.sort((a, b) => {
            const codeA = mortonCodes[a];
            const codeB = mortonCodes[b];
            return codeA < codeB ? -1 : codeA > codeB ? 1 : 0;
        });
        // Reorder points based on sorted Morton codes
        const sortedPoints = new Float32Array(points.length);

        for (let i = 0; i < numPoints; i++) {
            const oldOffset = indices[i] * pointSize;
            const newOffset = i * pointSize;
            sortedPoints.set(points.subarray(oldOffset, oldOffset + pointSize), newOffset);
        };

        return { sortedPoints, sortedIndices: indices };
    }

    computeMortonCodeXZ(
        x: number,
        z: number,
        bounds: Bounds
    ): number {
        const MAX_VALUE = (1 << 16); // 65535
        const safeNormalize = (v: number, min: number, max: number) => {
            const normalized = (v - min) / (max - min);
            return Math.min(normalized, 1.0 - 1e-6);
        };

        const normX = Math.floor(safeNormalize(x, bounds.min.x, bounds.max.x) * MAX_VALUE);
        const normZ = Math.floor(safeNormalize(z, bounds.min.z, bounds.max.z) * MAX_VALUE);
        return this.morton2D(normX, normZ); // Interleave bits for XZ only
    }

    morton2D(x: number, z: number): number {
        let morton = 0;
        for (let i = 0; i < 16; i++) {
            morton |= ((x >> i) & 1) << (2 * i) | ((z >> i) & 1) << (2 * i + 1);
        }
        return morton;
    }

}
