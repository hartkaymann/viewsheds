import { vec2, vec3 } from "gl-matrix";
import { Camera } from "./camera";
import Delaunator from "delaunator";
import { LASDecoder, LASFile } from "./laslaz";
import { MortonSorter, QuadTree } from "./optimization";

export class Scene {

    camera: Camera

    points: Float32Array;
    colors: Float32Array;
    indices: Uint32Array;
    triangleCount: number;
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
        // this.loadPLYFromURL("/model/galleon.ply");
        const url = "./model/80049_1525964_M-34-63-B-b-1-4-4-3.laz";
        await this.fetchAndProcessLASorLAZ(url);
    }

    async fetchAndProcessLASorLAZ(url: string) {
        try {
            console.log(`Fetching binary data from: ${url}`);

            // Fetch the LAZ/LAS file as an ArrayBuffer
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to load file: ${response.statusText}`);
            const arrayBuffer = await response.arrayBuffer();

            console.log("Successfully fetched file, initializing LAS/LAZ parser...");

            // Use LASFile from laslaz.js
            const lasFile = new LASFile(arrayBuffer);
            await lasFile.open();

            // Read the header
            const header = await lasFile.getHeader();
            console.log("Header Info:", header);

            // Read and process points
            const totalPoints = header.pointsCount;
            const pointsToExtract = 1000;
            console.log(`Total Points: ${totalPoints}`);

            const data = await lasFile.readData(totalPoints, 0, 1);
            console.log("Read Data:", data);

            const decoder = new LASDecoder(data.buffer, totalPoints, header);
            this.points = new Float32Array(pointsToExtract * 4);
            this.colors = new Float32Array(pointsToExtract * 4);

            let extractedCount = 0;
            const step = Math.floor(totalPoints / pointsToExtract);
            for (let i = 0; i < totalPoints && extractedCount < pointsToExtract; i += step) {
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
            // Close the file
            await lasFile.close();

            //Update camera position to focus on the loaded points
            const centerX = (header.maxs[0] - header.mins[0]) / 2;
            const centerY = (header.maxs[2] - header.mins[2]) / 2;
            const centerZ = (header.maxs[1] - header.mins[1]) / 2;
            this.camera.setPosition(vec3.fromValues(centerX, centerY, centerZ + (1 / header.scale[2]) * 10));
            this.camera.setTarget(vec3.fromValues(centerX, centerY, centerZ));

            let bounds = { x: 0, z: 0, width: header.maxs[0] - header.mins[0], height: header.maxs[1] - header.mins[1] }
            
            // Sort points 
            let sorter = new MortonSorter();
            let { sortedPoints, sortedIndices } = sorter.sort(this.points, { minX: bounds.x, minZ: bounds.z, maxX: bounds.width, maxZ: bounds.height });
            
            this.points = sortedPoints;
            this.colors = this.reorderColors(this.colors, sortedIndices);

            // Create quad tree
            this.tree = new QuadTree(bounds, 8);
            this.tree.assignPoints(this.points);

            // Triangulate
            const coords = new Float64Array(pointsToExtract * 2);
            for (let i = 0; i < pointsToExtract; i++) {
                coords[i * 2] = this.points[i * 4];         // x coordinate
                coords[i * 2 + 1] = this.points[i * 4 + 2]; // z coordinate
            }

            const delaunay = new Delaunator(coords);
            this.indices = new Uint32Array(delaunay.triangles);
            this.triangleCount = delaunay.triangles.length / 3;

            // TODO: Check if tree works, how do i access tree smart with ray position and load only those parts? 

        } catch (error) {
            console.error("Error loading LAS/LAZ file:", error);
        }
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