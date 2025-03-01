import { vec2, vec3 } from "gl-matrix";
import { Camera } from "./camera";
import Delaunator from "delaunator";
import { LASDecoder, LASFile } from "./laslaz";

export class Scene {

    camera: Camera

    points: Float32Array;
    colors: Float32Array;
    indices: Uint32Array;
    triangleCount: number;

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
            const pointsToExtract = 100;
            console.log(`Total Points: ${totalPoints}`);

            const data = await lasFile.readData(pointsToExtract, 0, 1);
            console.log("Read Data:", data);

            const decoder = new LASDecoder(data.buffer, totalPoints, header);
            this.points = new Float32Array(pointsToExtract * 4);
            this.colors = new Float32Array(pointsToExtract * 4);

            for (let i = 0; i < pointsToExtract; i++) {
                const point = decoder.getPoint(i);

                let position = vec3.create();
                vec3.multiply(position, point.position, header.scale);
                vec3.add(position, position, header.offset);

                // Normalize the position to be within 0 to 1 based on header maxs and mins
                position[0] = position[0] - header.mins[0];
                position[1] = position[1] - header.mins[1];
                position[2] = position[2] - header.mins[2];

                this.points[i * 4] = position[0];
                this.points[i * 4 + 1] = position[1];
                this.points[i * 4 + 2] = position[2];
                this.points[i * 4 + 3] = 1.0; // w

                let color = vec3.create();
                vec3.scale(color, point.color, 1 / 255);

                this.colors[i * 4] = color[0];
                this.colors[i * 4 + 1] = color[1];
                this.colors[i * 4 + 2] = color[2];
                this.colors[i * 4 + 3] = 1.0; // alpha
            }

            const coords = new Float64Array(pointsToExtract * 2);
            for (let i = 0; i < pointsToExtract; i++) {
                coords[i * 2] = this.points[i * 4];       // x coordinate
                coords[i * 2 + 1] = this.points[i * 4 + 1]; // y coordinate
            }

            const delaunay = new Delaunator(coords);
            this.indices = new Uint32Array(delaunay.triangles);
            this.triangleCount = delaunay.triangles.length / 3;

            // Close the file
            await lasFile.close();
        } catch (error) {
            console.error("Error loading LAS/LAZ file:", error);
        }
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