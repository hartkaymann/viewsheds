import { Renderer } from "../Renderer";
import { Utils } from "../Utils";

export class UIController {

    renderer: Renderer;

    constructor(renderer: Renderer) {
        this.renderer = renderer;    
    }

    init() {
        document.getElementById("raySampleInputs")?.addEventListener("change", this.handleUpdateRaySamples.bind(this));
        document.getElementById("originInputs")?.addEventListener("change", this.handleUpdateRayOrigin.bind(this));
        document.getElementById("thetaPhiInputs")?.addEventListener("change", this.handleUpdateThetaPhi.bind(this));
        document.getElementById("renderMode")?.addEventListener("change", this.handleUpdateRenderMode.bind(this));
        document.getElementById("clearPoints")?.addEventListener("click", this.handleClearVisibility.bind(this));
        document.getElementById("runNodes")?.addEventListener("click", this.handleRunComputePass.bind(this));
        document.getElementById("runPoints")?.addEventListener("click", this.handleRunCollision.bind(this));
        document.getElementById("renderPoints")?.addEventListener("change", this.handleRenderPointsChanged.bind(this));
        document.getElementById("renderMesh")?.addEventListener("change", this.handleRenderMeshChanged.bind(this));
        document.getElementById("renderRays")?.addEventListener("change", this.handleRenderRaysChanged.bind(this));
        document.getElementById("renderNodes")?.addEventListener("change", this.handleRenderNodesChanged.bind(this));

        this.handleUpdateRaySamples();
        this.handleUpdateRayOrigin();
        this.handleUpdateThetaPhi();
        this.handleUpdateRenderMode();
        this.handleRenderPointsChanged();
        this.handleRenderMeshChanged();
        this.handleRenderRaysChanged();
        this.handleRenderNodesChanged();
    }

    handleUpdateRaySamples() {
        const samplesX = parseInt((<HTMLInputElement>document.getElementById("samplesX")).value);
        const samplesY = parseInt((<HTMLInputElement>document.getElementById("samplesY")).value);
        this.renderer.updateRaySamples([samplesX, samplesY]);
    }

    handleUpdateRayOrigin() {
        const ox = parseFloat((<HTMLInputElement>document.getElementById("originX")).value);
        const oy = parseFloat((<HTMLInputElement>document.getElementById("originY")).value);
        const oz = parseFloat((<HTMLInputElement>document.getElementById("originZ")).value);
        this.renderer.updateRayOrigin([ox, oy, oz]);
    }

    handleUpdateThetaPhi() {
        const startTheta = parseFloat((<HTMLInputElement>document.getElementById("startTheta")).value);
        const endTheta = parseFloat((<HTMLInputElement>document.getElementById("endTheta")).value);
        const startPhi = parseFloat((<HTMLInputElement>document.getElementById("startPhi")).value);
        const endPhi = parseFloat((<HTMLInputElement>document.getElementById("endPhi")).value);
        this.renderer.updateThetaPhi([startTheta, endTheta], [startPhi, endPhi]);
    }

    handleUpdateRenderMode() {
        const renderMode = parseInt((<HTMLInputElement>document.getElementById("renderMode")).value);
        this.renderer.updateRenderMode(renderMode);
    }

    handleClearVisibility() {
        this.renderer.clearVisibility();
    }

    handleRunComputePass() {
        const sortNodesCheckbox = <HTMLInputElement>document.getElementById("sortNodes");
        const sortNodes = sortNodesCheckbox.checked;

        this.renderer.runComputePass(sortNodes);

        Utils.copyAndDisplayRayDebugData(this.renderer.device, this.renderer.bufferManager, this.renderer.raySamples, this.renderer.scene.tree.depth);
    }

    handleRunCollision() {
        this.renderer.runCollision();
    }

    handleRenderPointsChanged() {
        const renderPointsCheckbox = <HTMLInputElement>document.getElementById("renderPoints");
        const renderPoints = renderPointsCheckbox.checked;
        this.renderer.renderPoints = renderPoints;
    }

    handleRenderMeshChanged() {
        const renderMeshCheckbox = <HTMLInputElement>document.getElementById("renderMesh");
        const renderMesh = renderMeshCheckbox.checked;
        this.renderer.renderMesh = renderMesh;
    }

    handleRenderRaysChanged() {
        const renderRaysCheckbox = <HTMLInputElement>document.getElementById("renderRays");
        const renderRays = renderRaysCheckbox.checked;
        this.renderer.renderRays = renderRays;
    }

    handleRenderNodesChanged() {
        const renderNodesCheckbox = <HTMLInputElement>document.getElementById("renderNodes");
        const renderNodes = renderNodesCheckbox.checked;
        this.renderer.renderNodes = renderNodes;
    }

    updateButtonStates(disabled: boolean) {
        const runNodesButton = document.getElementById("runNodes") as HTMLButtonElement;
        if (runNodesButton) {
            runNodesButton.disabled = disabled;
        }
    
        const runPointsButton = document.getElementById("runPoints") as HTMLButtonElement;
        if (runPointsButton) {
            runPointsButton.disabled = disabled;
        }
    
        const runPanoramaButton = document.getElementById("runPanorama") as HTMLButtonElement;
        if (runPanoramaButton) {
            runPanoramaButton.disabled = disabled;
        }
    }
}