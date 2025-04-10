import { CollisionSystem } from "../CollisionSystem";
import { Controller } from "../Controller";
import { SceneSyncer } from "../SceneSyncer";
import { Utils } from "../Utils";
import { Viewport } from "../Viewport";

export class UIController {

    controller: Controller;
    viewport: Viewport;
    collision: CollisionSystem;

    constructor(controller: Controller) {
        this.controller = controller;
        this.viewport = controller.viewports;
        this.collision = controller.collisionSystem;
    }

    async init() {
        document.getElementById("raySampleInputs")?.addEventListener("change", this.handleUpdateRaySamples.bind(this));
        document.getElementById("originInputs")?.addEventListener("change", this.handleUpdateRayOrigin.bind(this));
        document.getElementById("thetaPhiInputs")?.addEventListener("change", this.handleUpdateThetaPhi.bind(this));
        document.getElementById("renderMode")?.addEventListener("change", this.handleUpdateRenderMode.bind(this));
        document.getElementById("clearPoints")?.addEventListener("click", this.handleClearVisibility.bind(this));
        document.getElementById("runNodes")?.addEventListener("click", this.handleNodeCollision.bind(this));
        document.getElementById("runPoints")?.addEventListener("click", this.handleRunPointCollision.bind(this));
        document.getElementById("runPanorama")?.addEventListener("click", this.handleRunPanorama.bind(this));
        document.getElementById("renderPoints")?.addEventListener("change", this.handleRenderPointsChanged.bind(this));
        document.getElementById("renderMesh")?.addEventListener("change", this.handleRenderMeshChanged.bind(this));
        document.getElementById("renderRays")?.addEventListener("change", this.handleRenderRaysChanged.bind(this));
        document.getElementById("renderNodes")?.addEventListener("change", this.handleRenderNodesChanged.bind(this));

        await this.handleUpdateRaySamples();
        await this.handleUpdateRayOrigin();
        await this.handleUpdateThetaPhi();
        this.handleUpdateRenderMode();
        this.handleRenderPointsChanged();
        this.handleRenderMeshChanged();
        this.handleRenderRaysChanged();
        this.handleRenderNodesChanged();
    }

    async handleUpdateRaySamples() {
        const samplesX = parseInt((<HTMLInputElement>document.getElementById("samplesX")).value);
        const samplesY = parseInt((<HTMLInputElement>document.getElementById("samplesY")).value);
        this.controller.updateRaySamples([samplesX, samplesY]);
        await this.collision.runGenerateRays();
    }

    async handleUpdateRayOrigin() {
        const ox = parseFloat((<HTMLInputElement>document.getElementById("originX")).value);
        const oy = parseFloat((<HTMLInputElement>document.getElementById("originY")).value);
        const oz = parseFloat((<HTMLInputElement>document.getElementById("originZ")).value);
        this.controller.scene.rays.origin = [ox, oy, oz];
        this.controller.updateRayOrigin([ox, oy, oz]);
        await this.collision.runGenerateRays();
    }

    async handleUpdateThetaPhi() {
        const startTheta = parseFloat((<HTMLInputElement>document.getElementById("startTheta")).value);
        const endTheta = parseFloat((<HTMLInputElement>document.getElementById("endTheta")).value);
        const startPhi = parseFloat((<HTMLInputElement>document.getElementById("startPhi")).value);
        const endPhi = parseFloat((<HTMLInputElement>document.getElementById("endPhi")).value);
        this.controller.scene.rays.theta = [startTheta, endTheta];
        this.controller.scene.rays.phi = [startPhi, endPhi];
        this.controller.updateThetaPhi([startTheta, endTheta], [startPhi, endPhi]);
        await this.collision.runGenerateRays();
    }

    handleUpdateRenderMode() {
        const renderMode = parseInt((document.getElementById("renderMode") as HTMLSelectElement).value, 10);
        this.viewport.updateRenderMode(renderMode);
    }

    handleClearVisibility() {
        this.viewport.clearVisibility();
    }

    handleNodeCollision() {
        const sortNodesCheckbox = <HTMLInputElement>document.getElementById("sortNodes");
        const sortNodes = sortNodesCheckbox.checked;

        this.collision.runNodeCollision(sortNodes);

        Utils.copyAndDisplayRayDebugData(this.controller.device, this.controller.bufferManager, this.controller.scene.rays.samples, this.controller.scene.tree.depth);
    }

    handleRunPointCollision() {
        this.collision.runPointCollision();
    }

    async handleRunPanorama() {
        const panoramaButton = document.getElementById("runPanorama") as HTMLButtonElement;

        if (!this.controller.collisionSystem.runningPanorama) {
            this.controller.collisionSystem.runningPanorama = true;
            panoramaButton.innerText = "Cancel Panorama";

            await this.controller.runPanoramaPass();

            this.controller.collisionSystem.runningPanorama = false;
            panoramaButton.innerText = "Run Panorama";
        } else {
            this.controller.collisionSystem.runningPanorama = false;

        }
    }

    handleRenderPointsChanged() {
        const renderPointsCheckbox = <HTMLInputElement>document.getElementById("renderPoints");
        const renderPoints = renderPointsCheckbox.checked;
        this.controller.renderSettings.points = renderPoints;
    }

    handleRenderMeshChanged() {
        const renderMeshCheckbox = <HTMLInputElement>document.getElementById("renderMesh");
        const renderMesh = renderMeshCheckbox.checked;
        this.controller.renderSettings.mesh = renderMesh;
    }

    handleRenderNodesChanged() {
        const renderNodesCheckbox = <HTMLInputElement>document.getElementById("renderNodes");
        const renderNodes = renderNodesCheckbox.checked;
        this.controller.renderSettings.nodes = renderNodes;
    }

    handleRenderRaysChanged() {
        const renderRaysCheckbox = <HTMLInputElement>document.getElementById("renderRays");
        const renderRays = renderRaysCheckbox.checked;
        this.controller.renderSettings.rays = renderRays;
    }

    setRunNodesButtonDisabled(disabled: boolean) {
        const runNodesButton = document.getElementById("runNodes") as HTMLButtonElement;
        if (runNodesButton) {
            runNodesButton.disabled = disabled;
        }
    }

    setRunPointsButtonDisabled(disabled: boolean) {
        const runPointsButton = document.getElementById("runPoints") as HTMLButtonElement;
        if (runPointsButton) {
            runPointsButton.disabled = disabled;
        }
    }

    setRunPanoramaButtonDisabled(disabled: boolean) {
        const runPanoramaButton = document.getElementById("runPanorama") as HTMLButtonElement;
        if (runPanoramaButton) {
            runPanoramaButton.disabled = disabled;
        }
    }
}