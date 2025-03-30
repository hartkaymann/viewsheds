import { Bounds } from "./types/types";

export interface SceneLoaderCallbacks {
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
                        treeData: msg.tree,
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

    startLoadPoints(url) {
        this.worker.postMessage({ type: "load-url", url });
    }

    startTreeBuild(points, bounds, depth) {
        this.worker.postMessage({ type: "build-tree", points, bounds, depth });
    }

    startTriangulation(points, tree, depth) {
        this.worker.postMessage({ type: "triangulate", points, tree, depth });
    }

    terminate() {
        this.worker.terminate();
    }

    setCallbacks(callbacks: Partial<SceneLoaderCallbacks>) {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }
}