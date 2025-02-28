import { vec2, vec3 } from "gl-matrix";
import { Camera } from "./camera";
import Delaunator from "delaunator";
import { LASFile } from "./laslaz.js";

export class Scene {

    camera: Camera

    points: Float32Array;
    indices: Uint32Array;
    triangleCount: number;

    constructor(camera: Camera) {
        this.camera = camera;

        this.init();
    }

    async init() {
        // this.loadPLYFromURL("/model/galleon.ply");
        await this.loadLASorLAZ("./model/80049_1525964_M-34-63-B-b-1-4-4-3.laz");
    }

    async loadLASorLAZ(url: string) {
        try {
            console.log(`Fetching binary data from: ${url}`);

            // Fetch the LAZ/LAS file as an ArrayBuffer using XMLHttpRequest method
            const buffer = await this.getBinary(url, (loaded, total) => {
                console.log(`Progress: ${(loaded / total) * 100}%`);
            });

            console.log("Successfully fetched file, initializing LAS/LAZ parser...");

            // Use LASFile from laslaz.js
            const lasFile = new LASFile(buffer);
            await lasFile.open();

            // Read the header
            const header = await lasFile.getHeader();
            console.log("Header Info:", header);

            // Read and process points
            const totalPoints = header.pointsCount;
            console.log(`Total Points: ${totalPoints}`);

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