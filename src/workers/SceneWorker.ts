import { vec3 } from "gl-matrix";
import { LASDecoder, LASFile } from "./laslaz";
import { MortonSorter, QuadTree } from "../Optimization";
import Delaunator from "delaunator";

type WorkerMessage =
    | { type: "load-points"; url: string }
    | { type: "build-tree"; points: Float32Array; bounds: Bounds; depth: number }
    | { type: "triangulate"; points: Float32Array; tree: any }; // tree must be serializable

export interface Bounds {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
}


self.onmessage = async (e) => {
    const msg = e.data;
    try {
        switch (msg.type) {
            case "load-points": {
                const { url } = msg;

                console.log(`Checking cache for: ${url}`);

                const cached = await loadFromCache(url);
                if (cached) {
                    const bounds = calculateBounds(cached.points);
                    postMessage({ type: "loaded", points: cached.points, colors: cached.colors, bounds });
                    return;
                }

                const { points: rawPoints, colors: rawColors } = await loadFromFile(url);
                const { points, colors, bounds } = sortPointCloud(rawPoints, rawColors);

                storeInCache(url, points, colors);
                postMessage({ type: "loaded", points, colors, bounds });
                return;
            }

            case "build-tree": {
                const { points, bounds, depth } = msg;
                const tree = await createQuadtree(points, bounds, depth);
                postMessage({ type: "tree-built", tree });
                return;
            }

            case "triangulate": {
                const { points, tree, depth } = msg;
                const result = performTriangulation(points, tree, depth);
                postMessage({ type: "triangulated", ...result });
                return;
            }
        }
    } catch (err) {
        postMessage({ type: "error", error: (err as Error).message });
    }
};

function loadFromCache(url: string): Promise<{ points: Float32Array; colors: Float32Array } | null> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("PointCloudCache", 1);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains("pointData")) {
                db.createObjectStore("pointData");
            }
        };

        request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction("pointData", "readonly");
            const store = transaction.objectStore("pointData");

            const pointsKey = url + "_points";
            const colorsKey = url + "_colors";

            console.log(`Checking cache for: ${pointsKey} and ${colorsKey}`);

            const pointsRequest = store.get(url + "_points");
            const colorsRequest = store.get(url + "_colors");

            pointsRequest.onsuccess = () => {
                colorsRequest.onsuccess = () => {
                    if (pointsRequest.result && colorsRequest.result) {
                        console.log(`Loaded from IndexedDB: ${pointsKey}, ${colorsKey}`);
                        resolve({
                            points: new Float32Array(pointsRequest.result),
                            colors: new Float32Array(colorsRequest.result)
                        });
                    } else {
                        console.warn(`No cache found for: ${pointsKey}, ${colorsKey}`);
                        resolve(null);
                    }
                };
            };

            transaction.onerror = () => reject(transaction.error);
        };

        request.onerror = () => reject(request.error);
    });
}

async function loadFromFile(url: string): Promise<{ points: Float32Array, colors: Float32Array }> {
    console.log(`Fetching binary data from: ${url}`);

    // Fetch the LAZ/LAS file as an ArrayBuffer
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load file: ${response.statusText}`);
    const arrayBuffer = await response.arrayBuffer();

    console.log("Successfully fetched file, initializing LAS/LAZ parser...");
    const lasFile = new LASFile(arrayBuffer);
    await lasFile.open();

    // Read the header
    const header = await lasFile.getHeader();
    console.log("Header Info:", header);

    // Read and process points
    const totalPoints = header.pointsCount;
    console.log(`Total Points: ${totalPoints}`);

    const data = await lasFile.readData(totalPoints, 0, 1);
    const decoder = new LASDecoder(data.buffer, totalPoints, header);

    const points = new Float32Array(totalPoints * 4);
    const colors = new Float32Array(totalPoints * 4);

    let extractedCount = 0;
    for (let i = 0; i < totalPoints; i++) {
        const point = decoder.getPoint(i);

        let position = vec3.create();
        vec3.multiply(position, point.position, header.scale);
        vec3.add(position, position, header.offset);

        position[0] = position[0] - header.mins[0];
        position[1] = position[1] - header.mins[1];
        position[2] = position[2] - header.mins[2];

        points[extractedCount * 4] = position[0];
        points[extractedCount * 4 + 1] = position[2]; // swap y and z
        points[extractedCount * 4 + 2] = position[1];
        points[extractedCount * 4 + 3] = 1.0; // w

        let color = vec3.create();
        vec3.scale(color, point.color, 1 / 255);

        colors[extractedCount * 4] = color[0];
        colors[extractedCount * 4 + 1] = color[1];
        colors[extractedCount * 4 + 2] = color[2];
        colors[extractedCount * 4 + 3] = 1.0; // alpha

        extractedCount++;
    }
    await lasFile.close();

    return { points, colors };
}

async function storeInCache(url: string, points: Float32Array, colors: Float32Array) {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("PointCloudCache", 1);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains("pointData")) {
                db.createObjectStore("pointData");
            }
        };

        request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction("pointData", "readwrite");
            const store = transaction.objectStore("pointData");

            store.put(points.buffer, url + "_points");
            store.put(colors.buffer, url + "_colors");

            transaction.oncomplete = () => resolve("Cached successfully!");
            transaction.onerror = () => reject(transaction.error);
        };

        request.onerror = () => reject(request.error);
    });
}

function calculateBounds(points: Float32Array): Bounds {
    const pointStride = 4;
    return points.reduce((acc, _, i) => {
        if (i % pointStride !== 0) return acc;
        const x = points[i];
        const y = points[i + 1];
        const z = points[i + 2];
        acc.min.x = Math.min(acc.min.x, x);
        acc.min.y = Math.min(acc.min.y, y);
        acc.min.z = Math.min(acc.min.z, z);
        acc.max.x = Math.max(acc.max.x, x);
        acc.max.y = Math.max(acc.max.y, y);
        acc.max.z = Math.max(acc.max.z, z);
        return acc;
    }, {
        min: { x: Infinity, y: Infinity, z: Infinity },
        max: { x: -Infinity, y: -Infinity, z: -Infinity }
    });
}

function sortPointCloud(points: Float32Array, colors: Float32Array): { points: Float32Array, colors: Float32Array, bounds: Bounds } {
    const bounds = this.calculateBounds(points);
    const sorter = new MortonSorter();
    const { sortedPoints, sortedIndices } = sorter.sort(points, bounds);
    const sortedColors = reorderFromSortedIndices(colors, sortedIndices);
    return { points: sortedPoints, colors: sortedColors, bounds };
}

function reorderFromSortedIndices(arr: Float32Array, sortedIndices: Uint32Array): Float32Array {
    const colorSize = 4;
    const sortedColors = new Float32Array(arr.length);
    sortedIndices.forEach((oldIndex, newIndex) => {
        const oldOffset = oldIndex * colorSize;
        const newOffset = newIndex * colorSize;
        sortedColors.set(arr.subarray(oldOffset, oldOffset + colorSize), newOffset);
    });
    return sortedColors;
}

function createQuadtree(points: Float32Array, bounds: Bounds, depth: number): ArrayBuffer {
    const tree = new QuadTree({
        pos: vec3.fromValues(bounds.min.x, bounds.min.y, bounds.min.z),
        size: vec3.fromValues(
            bounds.max.x - bounds.min.x,
            bounds.max.y - bounds.min.y,
            bounds.max.z - bounds.min.z
        ),
    }, depth);

    tree.assignIndices();
    tree.assignPoints(points);

    return tree.flatten();
}

function performTriangulation(
    points: Float32Array,
    treeData: ArrayBuffer,
    treeDepth: number
): {
    indices: Uint32Array;
    triangleCount: number;
    nodeToTriangles: Uint32Array;
} {
    const coords = new Float64Array(points.length / 2);
    for (let i = 0; i < points.length / 4; i++) {
        coords[i * 2] = points[i * 4];       // x
        coords[i * 2 + 1] = points[i * 4 + 2]; // z
    }

    const delaunay = new Delaunator(coords);
    const indices = new Uint32Array(delaunay.triangles);
    const triangleCount = indices.length / 3;

    const globalTriangleIndexBuffer: number[] = [];
    const tree = QuadTree.reconstruct(treeData, treeDepth);
    tree.assignTriangles(indices, points, globalTriangleIndexBuffer);
    const nodeToTriangles = new Uint32Array(globalTriangleIndexBuffer);

    return {
        indices,
        triangleCount,
        nodeToTriangles,
    };
}