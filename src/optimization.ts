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

    containsPoint(point: Float32Array): boolean {
        return (
            point[0] >= this.bounds.pos[0] &&
            point[0] < this.bounds.pos[0] + this.bounds.size[0] &&
            point[2] >= this.bounds.pos[2] &&
            point[2] < this.bounds.pos[2] + this.bounds.size[2]
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

    assignIndices(currentIndex: { value: number }) {
        this.index = currentIndex.value++;
        if (this.children) {
            for (const child of this.children) {
                child.assignIndices(currentIndex);
            }
        }
    }

    assignPoints(sortedPoints: Float32Array, startIndex: number, endIndex: number): void {
        if (this.children === null) { // Leaf
            this.startPointIndex = startIndex;
            this.pointCount = endIndex - startIndex;

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

        this.bounds.pos[1] = Infinity;
        this.bounds.size[1] = -Infinity;

        let currentStart = startIndex;
        for (const child of this.children) {
            let newEnd = currentStart;
            while (newEnd < endIndex) {
                const offset = newEnd * 4;
                const point = sortedPoints.subarray(offset, offset + 4);

                if (!child.containsPoint(point)) break;
                newEnd++;
            }

            if (newEnd > currentStart) {
                child.assignPoints(sortedPoints, currentStart, newEnd);

                this.bounds.pos[1] = Math.min(this.bounds.pos[1], child.bounds.pos[1]);
                this.bounds.size[1] = Math.max(
                    this.bounds.pos[1] + this.bounds.size[1],
                    child.bounds.pos[1] + child.bounds.size[1]
                ) - this.bounds.pos[1];

                currentStart = newEnd;
            }
        }
    }

    assignTriangles(
        triangles: Uint32Array,
        points: Float32Array,
        globalTriangleIndexBuffer: number[],
        relevantTriangles: Uint32Array | null = null
    ): number[] {
        let trianglesToProcess = relevantTriangles ?? triangles;

        if (this.children === null) { // Leaf node
            this.startTriangleIndex = globalTriangleIndexBuffer.length;
            this.triangleCount = 0;

            for (let i = 0; i < trianglesToProcess.length; i += 3) {
                let v0 = trianglesToProcess[i], v1 = trianglesToProcess[i + 1], v2 = trianglesToProcess[i + 2];

                let p0 = points.subarray(v0 * 4, v0 * 4 + 3);
                let p1 = points.subarray(v1 * 4, v1 * 4 + 3);
                let p2 = points.subarray(v2 * 4, v2 * 4 + 3);

                if (this.containsPoint(p0) || this.containsPoint(p1) || this.containsPoint(p2)) {
                    globalTriangleIndexBuffer.push(i / 3);
                    this.triangleCount++;
                }
            }
            return globalTriangleIndexBuffer;
        }

        let filteredTriangles: number[] = [];
        for (let i = 0; i < trianglesToProcess.length; i += 3) {
            let v0 = trianglesToProcess[i], v1 = trianglesToProcess[i + 1], v2 = trianglesToProcess[i + 2];

            let p0 = points.subarray(v0 * 4, v0 * 4 + 3);
            let p1 = points.subarray(v1 * 4, v1 * 4 + 3);
            let p2 = points.subarray(v2 * 4, v2 * 4 + 3);

            if (this.containsPoint(p0) || this.containsPoint(p1) || this.containsPoint(p2)) {
                filteredTriangles.push(v0, v1, v2);
            }
        }

        let filteredTriangleArray = new Uint32Array(filteredTriangles);
        for (const child of this.children) {
            child.assignTriangles(filteredTriangleArray, points, globalTriangleIndexBuffer, filteredTriangleArray);
        }

        return globalTriangleIndexBuffer;
    }
}

export class QuadTree {
    root: QuadTreeNode;

    constructor(bounds: AABB, depth: number) {
        this.root = new QuadTreeNode(bounds, depth);
    }

    assignPoints(sortedPoints: Float32Array): void {
        const pointSize = 4; // (x, y, z, w)
        this.root.assignPoints(sortedPoints, 0, sortedPoints.length / pointSize);
    }

    assignIndices() {
        this.root.assignIndices({ value: 0 });
    }

    assignTriangles(triangles: Uint32Array, points: Float32Array, globalTriangleIndexBuffer: number[]): void {
        this.root.assignTriangles(triangles, points, globalTriangleIndexBuffer);
    }

    flatten(): Uint32Array {
        const nodeList: QuadTreeNode[] = [];
        const queue: QuadTreeNode[] = [this.root];

        while (queue.length > 0) {
            const node = queue.shift()!;
            nodeList.push(node);

            if (node.children) {
                queue.push(...node.children);
            }
        }

        const nodeBuffer = new Float32Array(nodeList.length * 10);
        nodeList.forEach((node, i) => {
            const offset = i * 8;
            nodeBuffer.set(node.bounds.pos, offset);
            nodeBuffer.set(node.bounds.size, offset + 3);
            nodeBuffer[offset + 6] = node.startPointIndex;
            nodeBuffer[offset + 7] = node.pointCount;
            nodeBuffer[offset + 8] = node.startTriangleIndex;
            nodeBuffer[offset + 9] = node.triangleCount;
        });

        return new Uint32Array(nodeBuffer.buffer);
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
}

export class MortonSorter {

    constructor() { }

    sort(
        points: Float32Array,
        bounds: Bounds): { sortedPoints: Float32Array, sortedIndices: Uint32Array } {

        const pointSize = 4; // Each point has 4 components (x, y, z, w)
        const numPoints = points.length / pointSize;

        // Compute Morton code for each point
        let pointMortonPairs = [];
        for (let i = 0; i < numPoints; i++) {
            const offset = i * pointSize;
            const mortonCode = this.computeMortonCodeXZ(points.subarray(offset, offset + pointSize), bounds);
            pointMortonPairs.push({ index: i, mortonCode });
        }

        // Sort by Morton code
        pointMortonPairs.sort((a, b) => a.mortonCode - b.mortonCode);

        // Reorder points based on sorted Morton codes
        const sortedPoints = new Float32Array(points.length);
        const sortedIndices = new Uint32Array(numPoints);

        pointMortonPairs.forEach((pair, newIndex) => {
            const oldOffset = pair.index * pointSize;
            const newOffset = newIndex * pointSize;
            sortedPoints.set(points.subarray(oldOffset, oldOffset + pointSize), newOffset);
            sortedIndices[newIndex] = pair.index; // Store original index mapping

        });

        return { sortedPoints, sortedIndices };
    }

    computeMortonCodeXZ(
        point: Float32Array,
        bounds: Bounds
    ): number {
        const normX = Math.floor(1024 * (point[0] - bounds.min.x) / (bounds.max.x - bounds.min.x)); // Normalize X
        const normZ = Math.floor(1024 * (point[2] - bounds.min.z) / (bounds.max.z - bounds.min.z)); // Normalize Z
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