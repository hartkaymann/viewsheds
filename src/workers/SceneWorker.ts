/// <reference lib="webworker" />
/// <reference types="vite/client" />

import lazPerfUrl from 'laz-perf/lib/laz-perf.wasm?url';
import { createLazPerf } from "laz-perf";

import { vec3 } from "gl-matrix";
import { MortonSorter, QuadTree } from "../Optimization";
import { Bounds } from "../types/types";
import Delaunator from "delaunator";

let shouldShutdown = false;

const pointFormatReaders: Record<number, (dv: DataView) => {
    position: [number, number, number],
    intensity: number,
    classification: number,
    color?: [number, number, number]
}> = {
    0: (dv) => ({
        position: [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        intensity: dv.getUint16(12, true),
        classification: dv.getUint8(15)
    }),

    1: (dv) => ({
        position: [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        intensity: dv.getUint16(12, true),
        classification: dv.getUint8(15)
    }),

    2: (dv) => ({
        position: [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        intensity: dv.getUint16(12, true),
        classification: dv.getUint8(15),
        color: [dv.getUint16(20, true), dv.getUint16(22, true), dv.getUint16(24, true)]
    }),

    3: (dv) => ({
        position: [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        intensity: dv.getUint16(12, true),
        classification: dv.getUint8(15),
        color: [dv.getUint16(28, true), dv.getUint16(30, true), dv.getUint16(32, true)]
    }),

    // TODO: Add formats 5, 7, 8, 10
};

self.onmessage = async (e) => {
    const msg = e.data;
    try {
        switch (msg.type) {
            case "load-url": {
                const { url } = msg;
                await handleLoadAndRespond(url, async () => {
                    const response = await fetch(`${url}?nocache=${Date.now()}`);
                    if (!response.ok) throw new Error(`Failed to load file: ${response.statusText}`);
                    return response.arrayBuffer();
                });
                return;
            }

            case "load-arraybuffer": {
                const { name, buffer } = msg;
                await handleLoadAndRespond(name, async () => buffer);
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

async function handleLoadAndRespond(
    sourceId: string, // could be a URL or file name
    bufferProvider: () => Promise<ArrayBuffer>
) {
    if (shouldShutdown) return;

    console.log(`Checking cache for: ${sourceId}`);
    const cached = await loadFromCache(sourceId);
    if (cached) {
        const bounds = calculateBounds(cached.points);
        if (shouldShutdown) return;
        postMessage(
            { type: "loaded", ...cached, bounds },
            [cached.points.buffer, cached.colors?.buffer, cached.classification?.buffer].filter(Boolean)
        );
        return;
    }

    const buffer = await bufferProvider();
    const { points: rawPoints, colors: rawColors, classification: rawClassification } = await parseLASPoints(buffer);
    const { points, colors, classification, bounds } = sortPointCloud(rawPoints, rawColors, rawClassification);

    await storeInCache(sourceId, points, colors, classification);

    if (shouldShutdown) return;
    postMessage(
        { type: "loaded", points, colors, classification, bounds },
        [points.buffer, colors?.buffer, classification?.buffer].filter(Boolean)
    );
}

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
        formatId,
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

    const readPoint = pointFormatReaders[formatId];
    if (!readPoint) {
        throw new Error(`Unsupported point format: ${formatId}`);
    }

    for (let i = 0; i < pointCount; ++i) {
        laszip.getPoint(dataPtr);

        const raw = LazPerf.HEAPU8.subarray(dataPtr, dataPtr + pointDataRecordLength);
        const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        const point = readPoint(view);

        const [ix, iy, iz] = point.position;
        const x = ix * scale[0] + offset[0] - min[0];
        const y = iy * scale[1] + offset[1] - min[1];
        const z = iz * scale[2] + offset[2] - min[2];
        const w = 1.0;

        positions.set([x, z, y, w], i * 4); // Swapping Y and Z

        classifications[i] = point.classification;

        if (point.color) {
            const [rRaw, gRaw, bRaw] = point.color;
            const r = rRaw / 65535;
            const g = gRaw / 65535;
            const b = bRaw / 65535;
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

    let bit_7 = (pointDataRecordFormat & 0x80) >> 7;
    let bit_6 = (pointDataRecordFormat & 0x40) >> 6;
    const isCompressed = (bit_7 === 1 || bit_6 === 1);

    const formatId = pointDataRecordFormat & 0x3f;

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
        formatId,
        isCompressed,
        scale,
        offset,
        min,
        max,
    };
}