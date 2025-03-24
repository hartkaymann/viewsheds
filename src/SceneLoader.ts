import { Bounds } from "./workers/SceneWorker";

export interface SceneLoaderCallbacks {
    onPointsLoaded?: (data: {
        points: Float32Array;
        colors: Float32Array;
        bounds: Bounds;
    }) => void;
    onTreeBuilt?: (treeData: any) => void;
    onTriangulationDone?: (data: {
        indices: Uint32Array;
        triangleCount: number;
        nodeToTriangles: Uint32Array;
    }) => void;
    onError?: (error: string) => void;
}

export class SceneLoader {
    worker: Worker;
    private callbacks: SceneLoaderCallbacks = {};

    constructor(workerPath: string, callbacks?: SceneLoaderCallbacks) {
        const workerURL = new URL(workerPath, import.meta.url);
        this.worker = new Worker(workerURL, { type: "module" });
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
        this.worker.postMessage({ type: "load-points", url});
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