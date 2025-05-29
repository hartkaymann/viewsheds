import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { CollisionSystem } from "./CollisionSystem";
import { Gizmo } from "./Gizmo";
import { QuadTree } from "./Optimization";
import { PipelineManager } from "./PipelineManager";
import { Scene } from "./Scene";
import { SceneSyncer } from "./SceneSyncer";
import { UIController } from "./ui/UIController ";
import { Utils } from "./Utils";
import { Viewport } from "./Viewport";

export interface RenderPlan {
    transparent: boolean;
    mesh: boolean;
    nodes: boolean;
    rays: boolean;
    gizmo: boolean;
}

export interface RenderSettings {
    points: boolean;
    mesh: boolean;
    nodes: boolean;
    rays: boolean;
}

export class Controller {
    device: GPUDevice;

    scene: Scene;
    sync: SceneSyncer;

    collisionSystem: CollisionSystem;
    viewports: Viewport;

    ui: UIController | null = null;

    bufferManager: BufferManager;
    bindGroupsManager: BindGroupManager;

    renderSettings: RenderSettings =  {
        points: true,
        mesh: false,
        nodes: false,
        rays: true
    };

    canRender = {
        points: false,
        mesh: false,
        nodes: false,
        rays: true,
        gizmo: true
    }

    private prevTime = 0;
    private timeAccumulator = 0;
    private readonly timeStep = 1 / 60;
    private fps = 0;
    private fpsLastTime = 0;
    private frameCount = 0;
    private animationFrameId: number | null = null;
    private running = false;

    constructor(device: GPUDevice) {
        this.device = device;
        
        this.bufferManager = new BufferManager(this.device);
        this.bindGroupsManager = new BindGroupManager(this.device, this.bufferManager);
        
        this.scene = new Scene();
        this.sync = new SceneSyncer(this.scene, this.device, this.bufferManager, this.bindGroupsManager);

        this.viewports = new Viewport(this.device, this.scene, this.bufferManager, this.bindGroupsManager);
        this.collisionSystem = new CollisionSystem(this.device, this.scene, this.bufferManager, this.bindGroupsManager);
    }

    async init() {

        this.bufferManager.initBuffers([
            {
                name: "points",
                size: 16,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "colors",
                size: 16,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "classification",
                size: 4,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "indices",
                size: this.scene.indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "point_visibility",
                size: Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "node_visibility",
                size: QuadTree.leafNodes(this.scene.tree.depth) / 32 * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "ray_uniforms",
                size: 48,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                name: "rays",
                size: this.scene.rays.samples[0] * this.scene.rays.samples[1] * 2 * 4 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
            },
            {
                name: "nodes",
                size: QuadTree.totalNodes(this.scene.tree.depth) * QuadTree.BYTES_PER_NODE * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "point_to_node",
                size: this.scene.tree.root.pointCount * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "node_to_triangle",
                size: this.scene.nodeToTriangles.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "closest_hit",
                size: 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                data: new Uint32Array([0x7F7FFFFF, 0, 0, 0])
            },
            {
                name: "ray_nodes",
                size: this.scene.rays.samples[0] * this.scene.rays.samples[1] * QuadTree.noMaxNodesHit(this.scene.tree.depth) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "ray_node_counts",
                size: this.scene.rays.samples[0] * this.scene.rays.samples[1] * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "debug_distance",
                size: this.scene.rays.samples[0] * this.scene.rays.samples[1] * QuadTree.noMaxNodesHit(this.scene.tree.depth) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
        ]);

        this.bindGroupsManager.createLayout({
            name: "ray-uniforms",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "rays",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "find",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "sort",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "collision",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });



        this.bindGroupsManager.createGroup({
            name: "ray-uniforms",
            layoutName: "ray-uniforms",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("ray_uniforms") } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "rays",
            layoutName: "rays",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("rays") } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "find",
            layoutName: "find",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("ray_nodes") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("node_visibility") } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "sort",
            layoutName: "sort",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("ray_nodes") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("debug_distance") } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "collision",
            layoutName: "collision",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("points") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("indices") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("node_to_triangle") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("point_visibility") } },
            ]
        });

        await this.collisionSystem.init();
        await this.viewports.init();
    }

    start() {
        if (!this.running) {
            this.running = true;
            this.prevTime = performance.now() * 0.001;
            this.render();
        }
    }

    stop() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        this.running = false;
    }

    render = () => {
        if (!this.running) return;

        const currTime = performance.now();
        const deltaTime = currTime - this.prevTime;

        // Cap to 60 fps
        if (deltaTime >= 1000 / 60) {
            this.prevTime = currTime;

            if (!this.running) return;

            this.timeAccumulator += deltaTime;

            while (this.timeAccumulator >= this.timeStep) {
                this.timeAccumulator -= this.timeStep;
            }

            this.calculateFPS(currTime);

            // Parameter useless right now
            const plan = this.getRenderPlanFor(this.viewports);
            this.viewports.runRenderPass(plan);
        }

        this.animationFrameId = requestAnimationFrame(this.render);
    };

    async runPanoramaPass() {
        const startPhi = Math.PI / 3;
        const endPhi = 2 * Math.PI / 3;
        const step = Math.PI / 180; // 1 degree

        this.ui?.setRayInputsDisabled(true);

        for (let theta = 0; theta < 2 * Math.PI; theta += step) {
            if (!this.collisionSystem.runningPanorama) {
                Utils.showToast("Panorama stopped.", "info");
                break;
            }

            let startTheta = theta;
            let endTheta = theta + step;

            this.updateThetaPhi([startTheta, endTheta], [startPhi, endPhi]);
            await this.collisionSystem.runGenerateRays(false);
            await this.collisionSystem.runNodeCollision(true, false);
            await this.collisionSystem.runPointCollision(false);

            await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
        }

        this.ui?.setRayInputsDisabled(false);
    }

    async setPointData() {
        await this.sync.setPointData();
        this.canRender.points = true;
    }

    async setNodeData(resize: boolean = true) {
        await this.sync.setNodeData(resize);

        this.viewports.pipelineManager.update("render-nodes", {
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
            }
        });

        this.collisionSystem.pipelineManager.update("find-leaves", {
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            },
            codeConstants: {
                MAX_STACK_SIZE: 2 ** this.scene.tree.depth + 1
            }
        });

        this.collisionSystem.pipelineManager.update("collision", {
            constants: {
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            }
        });

        this.collisionSystem.updateRayWorkgroups();
        this.collisionSystem.updateRayPipelines();

        this.canRender.nodes = true;
    }

    async setMeshData() {
        await this.sync.setMeshData();
        this.canRender.mesh = true;
    }

    updateRaySamples(samples: [number, number] = [1, 1]) {
        const [oldX, oldY] = this.scene.rays.samples;

        this.scene.rays.samples = samples;
        this.sync.updateRaySamples(samples);

        if (oldX !== this.scene.rays.samples[0] || oldY !== this.scene.rays.samples[1]) {
            this.sync.resizeRayRelatedBuffers();
            this.collisionSystem.updateRayWorkgroups();
            this.collisionSystem.updateRayPipelines();
        }
    }

    updateRayOrigin(origin: [number, number, number] = [0, 0, 0]) {
        this.scene.rays.origin = origin;
        this.sync.updateRayOrigin(origin);
    }

    updateThetaPhi(theta: [number, number] = [0, 0], phi: [number, number] = [0, 0]) {
        this.scene.rays.theta = theta;
        this.scene.rays.phi = phi;
        this.sync.updateThetaPhi(theta, phi);

        this.ui?.updateThetaPhiInputs(theta, phi);
    }


    calculateFPS(currTime: number) {
        this.frameCount++;

        const elapsedTime = currTime - this.fpsLastTime;

        // Calculate FPS every second
        if (elapsedTime > 1000) {
            this.fps = this.frameCount / (elapsedTime / 1000);
            this.frameCount = 0;
            this.fpsLastTime = currTime;
        }

        const fpsLabel = document.getElementById("fps");
        if (fpsLabel) {
            fpsLabel.innerText = this.fps.toFixed(2);
        }
    }

    private getRenderPlanFor(viewport: Viewport): RenderPlan {
        return {
            transparent: this.canRender.points && this.renderSettings.points,
            mesh: this.canRender.mesh && this.renderSettings.mesh,
            nodes: this.canRender.nodes && this.renderSettings.nodes,
            rays: this.canRender.rays && this.renderSettings.rays,
            gizmo: this.canRender.gizmo,
        }
    }


    reset() {
        this.canRender = {
            points: false,
            mesh: false,
            nodes: false,
            rays: true,
            gizmo: true,
        }

        this.ui?.setRunNodesButtonDisabled(true);
        this.ui?.setRunPointsButtonDisabled(true);
        this.ui?.setRunPanoramaButtonDisabled(true);

        this.viewports.init();
    }

}