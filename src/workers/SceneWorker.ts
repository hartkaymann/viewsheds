/// <reference lib="webworker" />
/// <reference types="vite/client" />

import lazPerfUrl from 'laz-perf/lib/laz-perf.wasm?url';
import { createLazPerf } from "laz-perf";

import { vec3 } from "gl-matrix";
import { MortonSorter, QuadTree } from "../Optimization";
import { Bounds } from "../types/types";
import Delaunator from "delaunator";

let shouldShutdown = false;

self.onmessage = async (e) => {
    const msg = e.data;
    try {
        switch (msg.type) {
            case "load-url": {
                if (shouldShutdown) return;

                const { url } = msg;

                console.log(`Checking cache for: ${url}`);

                const cached = await loadFromCache(url);
                if (cached) {
                    const bounds = calculateBounds(cached.points);

                    if (shouldShutdown) return;
                    postMessage(
                        { type: "loaded", ...cached, bounds },
                        [cached.points.buffer, cached.colors?.buffer, cached.classification?.buffer].filter(Boolean)
                    );
                    return;
                }

                const { points: rawPoints, colors: rawColors, classification: rawClassification } = await loadFromFile(url);
                const { points, colors, classification, bounds } = sortPointCloud(rawPoints, rawColors, rawClassification);

                await storeInCache(url, points, colors, classification);


                if (shouldShutdown) return;
                self.postMessage(
                    { type: "loaded", points, colors, classification, bounds },
                    [points.buffer, colors.buffer, classification.buffer]
                );
                return;
            }

            case "build-tree": {
                if (shouldShutdown) return;

                const { points, bounds, depth } = msg;
                const tree = await createQuadtree(points, bounds, depth);

                if (shouldShutdown) return;
                postMessage({ type: "tree-built", tree: tree.flatten() });

                const result = performTriangulation(points, tree);

                if (shouldShutdown) return;
                postMessage({ type: "triangulated", ...result });
                return;
            }

            case "shutdown":
                shouldShutdown = true;
                self.close();
                break;
        }
    } catch (err) {
        postMessage({ type: "error", error: (err as Error).message });
    }
};

function loadFromCache(url: string): Promise<{
    points: Float32Array;
    colors: Float32Array;
    classification: Uint32Array;
} | null> {
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
            const classificationKey = url + "_classification";

            const getAsync = <T>(key: string): Promise<T | null> =>
                new Promise((res, rej) => {
                    const req = store.get(key);
                    req.onsuccess = () => res(req.result ?? null);
                    req.onerror = () => rej(req.error);
                });

            Promise.all([
                getAsync<ArrayBuffer>(pointsKey),
                getAsync<ArrayBuffer>(colorsKey),
                getAsync<ArrayBuffer>(classificationKey),
            ])
                .then(([pointsBuf, colorsBuf, classBuf]) => {
                    if (!pointsBuf) {
                        console.warn("Points not found in cache.");
                        resolve(null);
                        return;
                    }

                    const points = new Float32Array(pointsBuf);
                    const numPoints = points.length / 4;

                    // Fallback colors: white (1,1,1,1)
                    const colors = colorsBuf
                        ? new Float32Array(colorsBuf)
                        : (() => {
                            const fallback = new Float32Array(numPoints * 4);
                            for (let i = 0; i < numPoints; i++) {
                                fallback[i * 4 + 0] = 255;
                                fallback[i * 4 + 1] = 255;
                                fallback[i * 4 + 2] = 255;
                                fallback[i * 4 + 3] = 1;
                            }
                            return fallback;
                        })();

                    // Fallback classification: 0
                    const classification = classBuf
                        ? new Uint32Array(classBuf)
                        : new Uint32Array(numPoints).fill(0);

                    resolve({ points, colors, classification });
                })
                .catch(reject);
        };

        request.onerror = () => reject(request.error);
    });
}

async function loadFromFile(url: string): Promise<{
    points: Float32Array,
    colors: Float32Array,
    classification: Uint32Array
}> {
    console.log(`Fetching binary data from: ${url}`);

    const response = await fetch(`${url}?nocache=${Date.now()}`);
    if (!response.ok) throw new Error(`Failed to load file: ${response.statusText}`);

    const arrayBuffer = await response.arrayBuffer();
    console.log("Successfully fetched file, initializing LAS/LAZ parser...");

    return parseLASPoints(arrayBuffer);
}

async function parseLASPoints(arrayBuffer: ArrayBuffer): Promise<{
    points: Float32Array | null,
    colors: Float32Array | null,
    classification: Uint32Array | null
}> {
    const LazPerf = await createLazPerf({
        locateFile: () => lazPerfUrl
    });

    const laszip = new LazPerf.LASZip();

    const {
        pointCount,
        pointDataRecordLength,
        pointDataRecordFormat,
        scale,
        offset,
        min,
        max
    } = parseHeader(arrayBuffer);

    const uint8 = new Uint8Array(arrayBuffer);
    const filePtr = LazPerf._malloc(uint8.byteLength);
    LazPerf.HEAPU8.set(uint8, filePtr);
    laszip.open(filePtr, uint8.byteLength);

    const dataPtr = LazPerf._malloc(pointDataRecordLength);

    const positions = new Float32Array(pointCount * 4);
    const classifications = new Uint32Array(pointCount);
    const colors = new Float32Array(pointCount * 4);

    for (let i = 0; i < pointCount; ++i) {
        laszip.getPoint(dataPtr);

        const raw = LazPerf.HEAPU8.subarray(dataPtr, dataPtr + pointDataRecordLength);
        const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);

        const ix = view.getInt32(0, true);
        const iy = view.getInt32(4, true);
        const iz = view.getInt32(8, true);

        const x = ix * scale[0] + offset[0];
        const y = iy * scale[1] + offset[1];
        const z = iz * scale[2] + offset[2];
        const w = 1.0;

        positions.set([x, z, y, w], i * 4); // Swap y and z

        const classification = raw[15];
        classifications[i] = classification;

        // Optional color (only for formats with RGB: 2, 3, 5, 7, 8, 10)
        if (pointDataRecordFormat === 2 || pointDataRecordFormat === 3 || pointDataRecordFormat >= 5) {
            const r = view.getUint16(20, true) / 65535;
            const g = view.getUint16(22, true) / 65535;
            const b = view.getUint16(24, true) / 65535;
            const a = 1.0;
            colors.set([r, g, b, a], i * 4);
        }
    }

    // Clean up
    LazPerf._free(filePtr);
    LazPerf._free(dataPtr);
    laszip.delete();

    return {
        points: positions,
        colors,
        classification: classifications
    };
    // for (let i = 0; i < totalPoints; i++) {
    //     const point = decoder.getPoint(i);

    //     // Position
    //     const position = vec3.create();
    //     vec3.multiply(position, point.position, header.scale);
    //     vec3.add(position, position, header.offset);
    //     position[0] -= header.mins[0];
    //     position[1] -= header.mins[1];
    //     position[2] -= header.mins[2];

    //     points[extractedCount * 4] = position[0];
    //     points[extractedCount * 4 + 1] = position[2]; // swap y/z
    //     points[extractedCount * 4 + 2] = position[1];
    //     points[extractedCount * 4 + 3] = 1.0;

    //     if (point.color) {
    //         const color = vec3.create();
    //         vec3.scale(color, point.color, 1 / 255);
    //         colors[extractedCount * 4] = color[0];
    //         colors[extractedCount * 4 + 1] = color[1];
    //         colors[extractedCount * 4 + 2] = color[2];
    //         colors[extractedCount * 4 + 3] = 1.0;
    //     } else {
    //         colors.set([255, 255, 255, 1], extractedCount * 4);
    //     }

    //     if (point.classification) {
    //         classification[extractedCount] = point.classification;
    //     } else {
    //         classification[extractedCount] = 0;
    //     }

    //     extractedCount++;
    // }

    // await lasFile.close();
    // return { points, colors, classification };
}

async function storeInCache(
    url: string,
    points: Float32Array,
    colors?: Float32Array | null,
    classification?: Uint32Array | null
): Promise<void> {
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

            if (colors) {
                store.put(colors.buffer, url + "_colors");
            }

            if (classification) {
                store.put(classification.buffer, url + "_classification");
            }

            transaction.oncomplete = () => resolve();
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

function sortPointCloud(
    points: Float32Array,
    colors: Float32Array,
    classification: Uint32Array
): {
    points: Float32Array,
    colors: Float32Array,
    classification: Uint32Array,
    bounds: Bounds
} {
    const bounds = calculateBounds(points);
    const sorter = new MortonSorter();
    const { sortedPoints, sortedIndices } = sorter.sort(points, bounds);
    const sortedColors = reorderFromSortedIndices(colors, sortedIndices, 4);
    const sortedClassification = reorderFromSortedIndices(classification, sortedIndices, 1);
    return { points: sortedPoints, colors: sortedColors, classification: sortedClassification, bounds };
}

function reorderFromSortedIndices<T extends Float32Array | Uint32Array>(
    arr: T,
    sortedIndices: Uint32Array,
    tupleSize: number
): T {
    const sorted = new (arr.constructor as { new(length: number): T })(arr.length);
    sortedIndices.forEach((oldIndex, newIndex) => {
        const oldOffset = oldIndex * tupleSize;
        const newOffset = newIndex * tupleSize;
        sorted.set(arr.subarray(oldOffset, oldOffset + tupleSize), newOffset);
    });
    return sorted;
}

function createQuadtree(points: Float32Array, bounds: Bounds, depth: number): QuadTree {
    const tree = new QuadTree({
        pos: vec3.fromValues(bounds.min.x, bounds.min.y, bounds.min.z),
        size: vec3.fromValues(
            bounds.max.x - bounds.min.x,
            bounds.max.y - bounds.min.y,
            bounds.max.z - bounds.min.z
        ),
    }, depth);

    tree.assignPoints(points);

    return tree;
}

function performTriangulation(
    points: Float32Array,
    tree: QuadTree
): {
    indices: Uint32Array,
    triangleCount: number,
    treeData: ArrayBuffer,
    nodeToTriangles: Uint32Array
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
    tree.assignTriangles(indices, points, globalTriangleIndexBuffer);
    const nodeToTriangles = new Uint32Array(globalTriangleIndexBuffer);

    return {
        indices,
        triangleCount,
        treeData: tree.flatten(),
        nodeToTriangles,
    };

}

function parseHeader(arrayBuffer: ArrayBuffer) {
    const view = new DataView(arrayBuffer);

    const pointCount = view.getUint32(107, true);
    const pointDataRecordLength = view.getUint16(105, true);
    const pointDataRecordFormat = view.getUint8(104);

    const scale = [
        view.getFloat64(131, true),
        view.getFloat64(139, true),
        view.getFloat64(147, true),
    ];
    const offset = [
        view.getFloat64(155, true),
        view.getFloat64(163, true),
        view.getFloat64(171, true),
    ];
    const max = [
        view.getFloat64(179, true),
        view.getFloat64(195, true),
        view.getFloat64(211, true),
    ]
    const min = [
        view.getFloat64(187, true),
        view.getFloat64(203, true),
        view.getFloat64(219, true),
    ]

    return {
        pointCount,
        pointDataRecordLength,
        pointDataRecordFormat,
        scale,
        offset,
        min,
        max,
    };
}