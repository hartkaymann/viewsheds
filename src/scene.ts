import { vec2, vec3 } from "gl-matrix";
import { Camera } from "./camera";
import Delaunator from "delaunator";
import { LASDecoder, LASFile } from "./laslaz";
import { MortonSorter, QuadTree } from "./optimization";
import { Bounds } from "./types/types";

export class Scene {

    camera: Camera

    points: Float32Array;
    colors: Float32Array;
    indices: Uint32Array;
    nodeToTriangles: Uint32Array;

    triangleCount: number;
    bounds: Bounds;
    tree: QuadTree;

    gizmo: Float32Array = new Float32Array([
        0, 0, 0, 1, 1, 0, 0, 1,
        0, 0, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 1, 0, 0, 1, 1,
    ]);

    constructor(camera: Camera) {
        this.camera = camera;
    }

    async init() {
        const url = "./model/80049_1525964_M-34-63-B-b-1-4-4-3.laz";
        await this.fetchAndProcessLASorLAZ(url);

        this.focusCameraOnPointCloud();
    }

    async fetchAndProcessLASorLAZ(url: string) {
        try {
            console.log(`Checking cache for: ${url}`);

            const cachedData = await this.loadFromCache(url);
            if (cachedData) {
                console.log("Loaded from cache!");
                this.points = cachedData.points;
                this.colors = cachedData.colors;
            } else {
                console.log("No cache found. Fetching from file...");
                await this.loadFromFile(url);
                this.sortPointsCloud();

                this.storeInCache(url, this.points, this.colors).catch(err => {
                    console.error("Failed to store data in cache:", err);
                });
            }

            this.processPointCloud();
        } catch (error) {
            console.error("Error loading LAS/LAZ file:", error);
        }
    }

    async loadFromCache(url: string): Promise<{ points: Float32Array; colors: Float32Array } | null> {
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

    async loadFromFile(url: string) {
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

        this.points = new Float32Array(totalPoints * 4);
        this.colors = new Float32Array(totalPoints * 4);

        let extractedCount = 0;
        for (let i = 0; i < totalPoints; i++) {
            const point = decoder.getPoint(i);

            let position = vec3.create();
            vec3.multiply(position, point.position, header.scale);
            vec3.add(position, position, header.offset);

            position[0] = position[0] - header.mins[0];
            position[1] = position[1] - header.mins[1];
            position[2] = position[2] - header.mins[2];

            this.points[extractedCount * 4] = position[0];
            this.points[extractedCount * 4 + 1] = position[2]; // swap y and z
            this.points[extractedCount * 4 + 2] = position[1];
            this.points[extractedCount * 4 + 3] = 1.0; // w

            let color = vec3.create();
            vec3.scale(color, point.color, 1 / 255);

            this.colors[extractedCount * 4] = color[0];
            this.colors[extractedCount * 4 + 1] = color[1];
            this.colors[extractedCount * 4 + 2] = color[2];
            this.colors[extractedCount * 4 + 3] = 1.0; // alpha

            extractedCount++;
        }
        await lasFile.close();
    }

    async storeInCache(url: string, points: Float32Array, colors: Float32Array) {
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

    sortPointsCloud() {
        this.bounds = this.calculateBounds(this.points);
        let sorter = new MortonSorter();
        let { sortedPoints, sortedIndices } = sorter.sort(this.points, this.bounds);
        this.points = sortedPoints;
        this.colors = this.reorderColors(this.colors, sortedIndices);
    }

    processPointCloud() {
        // Find accurate bounds
        const bounds = this.bounds ?? (this.bounds = this.calculateBounds(this.points));

        // Create Quadtree
        this.tree = new QuadTree({
            pos: vec3.fromValues( bounds.min.x, bounds.min.y, bounds.min.z),
            size: vec3.fromValues(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z)
        }, 6);
        this.tree.assignIndices();
        this.tree.assignPoints(this.points);

        // Perform Delaunay Triangulation
        this.performTriangulation();
    }

    calculateBounds(points: Float32Array): Bounds {
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

    performTriangulation() {
        const coords = new Float64Array(this.points.length / 2);
        for (let i = 0; i < this.points.length / 4; i++) {
            coords[i * 2] = this.points[i * 4];
            coords[i * 2 + 1] = this.points[i * 4 + 2];
        }
        const delaunay = new Delaunator(coords);
        this.indices = new Uint32Array(delaunay.triangles);
        this.triangleCount = delaunay.triangles.length / 3;

        // Map nodes to triangles
        let globalTriangleIndexBuffer: number[] = [];
        this.tree.assignTriangles(this.indices, this.points,globalTriangleIndexBuffer);
        this.nodeToTriangles = new Uint32Array(globalTriangleIndexBuffer);

    }

    reorderColors(colors: Float32Array, sortedIndices: Uint32Array): Float32Array {
        const colorSize = 4; // (r, g, b, a)
        const sortedColors = new Float32Array(colors.length);

        sortedIndices.forEach((oldIndex, newIndex) => {
            const oldOffset = oldIndex * colorSize;
            const newOffset = newIndex * colorSize;
            sortedColors.set(colors.subarray(oldOffset, oldOffset + colorSize), newOffset);
        });

        return sortedColors;
    }

    focusCameraOnPointCloud() {
        const bounds = this.bounds;

        const centerX = (bounds.min.x + bounds.max.x) / 2;
        const centerY = (bounds.min.y + bounds.max.y) / 2;
        const centerZ = (bounds.min.z + bounds.max.z) / 2;

        // Move the camera back along the Z-axis to fit the whole cloud in view
        const distance = Math.max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z) * 1.5;

        this.camera.setPosition(vec3.fromValues(centerX, centerY, centerZ + distance));
        this.camera.setTarget(vec3.fromValues(centerX, centerY, centerZ));
    }

    async getBinary(url: string, progressCallback: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
        return new Promise((resolve, reject) => {
            const oReq = new XMLHttpRequest();
            oReq.open("GET", url, true);
            oReq.responseType = "arraybuffer"; // Ensure binary data is correctly retrieved

            oReq.onprogress = function (e) {
                if (e.lengthComputable) {
                    progressCallback(e.loaded, e.total); // Update progress
                }
            };

            oReq.onload = function () {
                if (oReq.status === 200) {
                    console.log("Response Headers:", oReq.getAllResponseHeaders());
                    resolve(oReq.response); // Correctly resolve as ArrayBuffer
                } else {
                    reject(new Error(`Failed to fetch file: ${oReq.statusText}`));
                }
            };

            oReq.onerror = function (err) {
                reject(new Error("Network error while fetching binary data"));
            };

            oReq.send();
        });
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