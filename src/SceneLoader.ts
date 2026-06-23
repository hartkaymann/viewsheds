import { Bounds } from "./types/types";

export interface LasHeaderSummary {
    fileName: string;
    fileSize: number;
    versionMajor: number;
    versionMinor: number;
    pointCount: number;
    formatId: number;
    isCompressed: boolean;
    pointDataRecordLength: number;
    min: number[];
    max: number[];
}

export interface SceneLoaderCallbacks {
    onLoadStart?: () => void;
    onHeader?: (summary: LasHeaderSummary) => void;
    onProgress?: (stage: string, value: number) => void;
    onPointsLoaded?: (data: {
        points: Float32Array,
        colors: Float32Array,
        classification: Uint32Array,
        bounds: Bounds;
    }) => void;
    onTreeBuilt?: (treeData: ArrayBuffer) => void;
    onTriangulationDone?: (data: {
        indices: Uint32Array,
        triangleCount: number,
        treeData: ArrayBuffer,
        nodeToTriangles: Uint32Array;
    }) => void;
    onError?: (error: string) => void;
}

export class SceneLoader {
    worker: Worker;
    private callbacks: SceneLoaderCallbacks = {};

    constructor(WorkerConstructor: new () => Worker, callbacks?: SceneLoaderCallbacks) {
        this.worker = new WorkerConstructor();
        this.callbacks = callbacks ?? {};
        this.worker.onmessage = (e: MessageEvent<any>) => {
            const msg = e.data;

            switch (msg.type) {
                case "header":
                    this.callbacks.onHeader?.(msg.summary);
                    break;

                case "progress":
                    this.callbacks.onProgress?.(msg.stage, msg.value);
                    break;

                case "loaded":
                    this.callbacks.onPointsLoaded?.(msg);
                    break;

                case "tree-built":
                    this.callbacks.onTreeBuilt?.(msg.tree);
                    break;

                case "triangulated":
                    this.callbacks.onTriangulationDone?.({
                        indices: msg.indices,
                        triangleCount: msg.triangleCount,
                        treeData: msg.treeData,
                        nodeToTriangles: msg.nodeToTriangles,
                    });
                    break;

                case "error":
                    console.error("SceneWorker error:", msg.error);
                    this.callbacks.onError?.(msg.error);
                    break;
            }
        };
    }

    peekHeader(buffer: ArrayBuffer, name: string) {
        this.worker.postMessage({ type: "peek-header", buffer, name }, [buffer]);
    }

    confirmLoad() {
        this.callbacks.onLoadStart?.();
        this.worker.postMessage({ type: "confirm-load" });
    }

    startTreeBuild(points, bounds, depth) {
        this.worker.postMessage({ type: "build-tree", points, bounds, depth });
    }

    shutdown() {
        this.worker.postMessage({ type: "shutdown" });
        setTimeout(() => {
            this.worker.terminate();
        }, 50);
    }

    setCallbacks(callbacks: Partial<SceneLoaderCallbacks>) {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }
}