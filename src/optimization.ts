import { vec2 } from "gl-matrix";

interface AABB {
    x: number;
    z: number;
    width: number;
    height: number
}

interface Triangle {
    index: number;
    center: vec2;
}

class QuadTreeNode {
    children: QuadTreeNode[] | null;
    bounds: AABB;
    startIndex: number;
    count: number;
    index: number;

    constructor(bounds: AABB, depth: number) {
        this.children = null;
        this.bounds = bounds;
        this.startIndex = -1;
        this.count = 0;
        this.index = -1;

        if (depth > 0) {
            const halfW = bounds.width / 2;
            const halfH = bounds.height / 2;

            this.children = [
                new QuadTreeNode({ x: bounds.x, z: bounds.z, width: halfW, height: halfH }, depth - 1),
                new QuadTreeNode({ x: bounds.x + halfW, z: bounds.z, width: halfW, height: halfH }, depth - 1),
                new QuadTreeNode({ x: bounds.x, z: bounds.z + halfH, width: halfW, height: halfH }, depth - 1),
                new QuadTreeNode({ x: bounds.x + halfW, z: bounds.z + halfH, width: halfW, height: halfH }, depth - 1)
            ]
        }
    }

    containsPoint(point: Float32Array): boolean {
        return (
            point[0] >= this.bounds.x &&
            point[0] < this.bounds.x + this.bounds.width &&
            point[2] >= this.bounds.z &&
            point[2] < this.bounds.z + this.bounds.height
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
}

export class QuadTree {
    root: QuadTreeNode;

    constructor(bounds: AABB, depth: number) {
        this.root = new QuadTreeNode(bounds, depth);
    }

    assignPoints(sortedPoints: Float32Array): void {
        const pointSize = 4; // (x, y, z, w)
        this.assignPointsRecursive(this.root, sortedPoints, 0, sortedPoints.length / pointSize);
    }

    assignIndices() {
        this.root.assignIndices({ value: 0 });
    }

    private assignPointsRecursive(node: QuadTreeNode, sortedPoints: Float32Array, startIndex: number, endIndex: number): void {
        if (node.children === null) { // Leaf
            node.startIndex = startIndex;
            node.count = endIndex - startIndex;
            return;
        }

        node.startIndex = startIndex;
        node.count = 0;

        let currentStart = startIndex;
        for (const child of node.children) {
            let newEnd = currentStart;
            while (newEnd < endIndex) {
                const offset = newEnd * 4;
                const point = sortedPoints.subarray(offset, offset + 4);

                if (!child.containsPoint(point)) break;
                newEnd++;
            }

            if (newEnd > currentStart) {
                // Assign points to child
                this.assignPointsRecursive(child, sortedPoints, currentStart, newEnd);

                // Accumulate child counts
                node.count += child.count;

                currentStart = newEnd;
            }
        }
    }

    flatten(): Uint32Array {
        const nodeList: QuadTreeNode[] = [];

        this.root.traverse((node) => {
            nodeList.push(node);
        });

        const nodeBuffer = new Uint32Array(nodeList.length * 5);
        nodeList.forEach((node, i) => {
            nodeBuffer[i * 4] = node.bounds.x;
            nodeBuffer[i * 4 + 1] = node.bounds.z;
            nodeBuffer[i * 4 + 2] = node.bounds.width;
            nodeBuffer[i * 4 + 2] = node.bounds.height;
            nodeBuffer[i * 4 + 3] = node.index;
        });

        return nodeBuffer;
    }

    mapPointsToNodes = (): Uint32Array => {
        const pointToNodeBuffer = new Uint32Array(this.root.count); // One index per point

        this.root.traverse(node => {
            if (node.children === null) { // Only assign leaf nodes
                for (let i = 0; i < node.count; i++) {
                    pointToNodeBuffer[node.startIndex + i] = node.index; // Assign node ID
                }
            }
        });

        return pointToNodeBuffer;
    };

    static getPointsInNode(node: QuadTreeNode, sortedPoints: Float32Array): Float32Array {
        const pointSize = 4;
        return sortedPoints.subarray(node.startIndex * pointSize, (node.startIndex + node.count) * pointSize);
    }
}

export class MortonSorter {

    constructor() {

    }

    sort(
        points: Float32Array,
        bounds: { minX: number, minZ: number, maxX: number, maxZ: number }): { sortedPoints: Float32Array, sortedIndices: Uint32Array } {

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
        bounds: { minX: number, minZ: number, maxX: number, maxZ: number }
    ): number {
        const normX = Math.floor(1024 * (point[0] - bounds.minX) / (bounds.maxX - bounds.minX)); // Normalize X
        const normZ = Math.floor(1024 * (point[2] - bounds.minZ) / (bounds.maxZ - bounds.minZ)); // Normalize Z
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