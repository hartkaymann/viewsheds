import find_leaves_src from "./shaders/find-leaves.wgsl"
import bitonic_sort_src from "./shaders/bitonic-sort.wgsl"

import wireframe_src from "./shaders/wireframe.wgsl"
import points_src from "./shaders/points.wgsl"
import rays_src from "./shaders/rays.wgsl"
import gizmo_src from "./shaders/gizmo.wgsl"
import nodes_src from "./shaders/nodes.wgsl"

import { Scene } from "./scene";
import { mat3, mat4, vec3 } from "gl-matrix";
import { WorkgroupLayout, WorkgroupLimits, WorkgroupStrategy } from "./types/types"
import { BufferManager } from "./BufferManager"
import { QuadTree } from "./optimization"
import { data } from "jquery"
import { BindGroupManager } from "./BindGroupsManager"


export class Renderer {

    canvas: HTMLCanvasElement

    // Device/Context objects
    adapter: GPUAdapter;
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    renderPassDescriptor: GPURenderPassDescriptor;

    //Assets
    depth_buffer: GPUTexture;
    depth_buffer_view: GPUTextureView;

    // Pipeline objects
    pipelineFindLeaves: GPUComputePipeline;
    pipelineBitonicSort: GPUComputePipeline;

    pipelineRenderPoints: GPURenderPipeline;
    pipelineRenderWireframe: GPURenderPipeline;
    pipelineRenderRays: GPURenderPipeline;
    pipelineRenderGizmo: GPURenderPipeline;
    pipelineRenderNodes: GPURenderPipeline;

    raySamples: Uint32Array = new Uint32Array([8, 8]);

    // Managers
    bufferManager: BufferManager;
    bindGroupsManager: BindGroupManager;

    // Others
    workgroupLayoutGrid: WorkgroupLayout;
    workgroupLayoutSort: WorkgroupLayout;

    // Scene to render
    scene: Scene

    // Time
    prevTime = 0;
    timeAccumulator = 0;
    readonly timeStep = 1 / 60;
    fps = 0;  // Frame rate value
    fpsLastTime = 0;  // Last time we calculated FPS
    frameCount = 0;  // Frames since the last FPS calculation

    constructor(canvas: HTMLCanvasElement, scene: Scene) {
        this.canvas = canvas;
        this.scene = scene;
    }

    async init() {
        this.device = await this.createDevice();

        this.context = <GPUCanvasContext>this.canvas.getContext("webgpu");
        this.format = "bgra8unorm";

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: "opaque"
        });

        this.bufferManager = new BufferManager(this.device);
        this.bufferManager.initBuffers([
            {
                name: "points",
                size: this.scene.points.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                data: this.scene.points
            },
            {
                name: "colors",
                size: this.scene.colors.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                data: this.scene.colors
            },
            {
                name: "indices",
                size: this.scene.indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                data: this.scene.indices
            },
            {
                name: "point_visibility",
                size: Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "node_visibility",
                size: Math.pow(4, this.scene.tree.depth) / 32 * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "comp_uniforms",
                size: 48,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                name: "vs_uniforms",
                size: 192,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                name: "rays",
                size: this.raySamples[0] * this.raySamples[1] * 2 * 4 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
            },
            {
                name: "gizmo_vertices",
                size: this.scene.gizmo.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                data: this.scene.gizmo
            },
            {
                name: "gizmo_uniforms",
                size: 192,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                name: "render_mode",
                size: 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                name: "nodes",
                size: QuadTree.totalNodes(this.scene.tree.depth) * QuadTree.BYTES_PER_NODE * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                data: this.scene.tree.flatten()
            },
            {
                name: "point_to_node",
                size: this.scene.tree.root.pointCount * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                data: this.scene.tree.mapPointsToNodes()
            },
            {
                name: "node_to_triangle",
                size: this.scene.nodeToTriangles.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                data: this.scene.nodeToTriangles
            },
            {
                name: "closest_hit",
                size: 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                data: new Uint32Array([0x7F7FFFFF, 0, 0, 0])
            },
            {
                name: "ray_nodes",
                size: this.raySamples[0] * this.raySamples[1] * 128 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "ray_node_counts",
                size: this.raySamples[0] * this.raySamples[1] * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
        ]);

        this.bindGroupsManager = new BindGroupManager(this.device);
        this.bindGroupsManager.createLayout({
            name: "find",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "sort",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "render",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "gizmo",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "points",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            ]
        });

        this.bindGroupsManager.createLayout({
            name: "nodes",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "find",
            layoutName: "find",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("rays") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("node_visibility") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("comp_uniforms") } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "sort",
            layoutName: "sort",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("rays") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("comp_uniforms") } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "render",
            layoutName: "render",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("points") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("indices") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("point_visibility") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("vs_uniforms") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("rays") } }
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "gizmo",
            layoutName: "gizmo",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("gizmo_uniforms") } },
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "points",
            layoutName: "points",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("render_mode") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("nodes") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("point_to_node") } }
            ]
        });

        this.bindGroupsManager.createGroup({
            name: "nodes",
            layoutName: "nodes",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("vs_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("nodes") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("node_visibility") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } }
            ]
        });

        const pipeline_layout_render = this.device.createPipelineLayout({
            label: 'pipeline-layout-render',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["render"])
        });
        const pipeline_layout_points = this.device.createPipelineLayout({
            label: 'pipeline-layout-points',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["render", "points"])
        });
        const pipeline_layout_gizmo = this.device.createPipelineLayout({
            label: 'pipeline-layout-gizmo',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["gizmo"])
        });
        const pipeline_layout_nodes = this.device.createPipelineLayout({
            label: 'pipeline-layout-nodes',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["nodes"])
        });

        this.setupComputePipelines();

        const pointShaderModule = this.device.createShaderModule({ code: points_src });
        this.pipelineRenderPoints = this.device.createRenderPipeline({
            label: 'render-pipeline',
            layout: pipeline_layout_points,
            vertex: {
                module: pointShaderModule,
                entryPoint: 'main',
                buffers: [
                    {
                        arrayStride: 4 * 4, // 4 floats @ 4 bytes
                        attributes: [
                            {
                                shaderLocation: 0, // Position
                                offset: 0,
                                format: 'float32x4'
                            }
                        ]
                    },
                    {
                        arrayStride: 4 * 4, // 4 floats @ 4 bytes
                        attributes: [
                            {
                                shaderLocation: 1, // Color
                                offset: 0,
                                format: 'float32x4'
                            }
                        ]
                    }
                ]
            },
            fragment: {
                module: pointShaderModule,
                entryPoint: 'main_fs',
                targets: [
                    {
                        format: 'bgra8unorm',
                        blend: {
                            color: {
                                srcFactor: 'src-alpha',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add'
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add'
                            }
                        }
                    }
                ]
            },
            primitive: {
                topology: 'point-list'
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        const wireframeShaderModule = this.device.createShaderModule({ code: wireframe_src });
        this.pipelineRenderWireframe = this.device.createRenderPipeline({
            label: 'render-pipeline',
            layout: pipeline_layout_render,
            vertex: {
                module: wireframeShaderModule,
                entryPoint: 'main',
            },
            fragment: {
                module: wireframeShaderModule,
                entryPoint: 'main_fs',
                targets: [
                    { format: 'bgra8unorm' }
                ]
            },
            primitive: {
                topology: 'line-strip',
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        const rayShaderModule = this.device.createShaderModule({ code: rays_src });
        this.pipelineRenderRays = this.device.createRenderPipeline({
            label: 'render-pipeline-rays',
            layout: pipeline_layout_render,
            vertex: {
                module: rayShaderModule,
                entryPoint: 'main_vs',
            },
            fragment: {
                module: rayShaderModule,
                entryPoint: 'main_fs',
                targets: [
                    { format: 'bgra8unorm' }
                ]
            },
            primitive: {
                topology: 'line-list'
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        const gizmoShaderModule = this.device.createShaderModule({ code: gizmo_src });
        this.pipelineRenderGizmo = this.device.createRenderPipeline({
            label: 'render-pipeline-gizmo',
            layout: pipeline_layout_gizmo,
            vertex: {
                module: gizmoShaderModule,
                entryPoint: 'main',
                buffers: [
                    {
                        arrayStride: 4 * 4,
                        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }],
                    }
                ],
            },
            fragment: {
                module: gizmoShaderModule,
                entryPoint: 'main_fs',
                targets: [
                    { format: 'bgra8unorm' }
                ]
            },
            primitive: {
                topology: 'line-strip',
            }
        });

        const nodesShaderModule = this.device.createShaderModule({ code: nodes_src });
        this.pipelineRenderNodes = this.device.createRenderPipeline({
            label: 'render-pipeline-nodes',
            layout: pipeline_layout_nodes,
            vertex: {
                module: nodesShaderModule,
                entryPoint: 'main',
                constants: {
                    TREE_DEPTH: this.scene.tree.depth,
                    BLOCK_SIZE: 128,
                }
            },
            fragment: {
                module: nodesShaderModule,
                entryPoint: 'main_fs',
                targets: [{
                    format: 'bgra8unorm',
                    blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "one",
                            dstFactor: "zero",
                            operation: "add",
                        },
                    },
                    writeMask: GPUColorWrite.ALL,
                }]
            },
            primitive: {
                topology: 'line-list',
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        this.depth_buffer = this.device.createTexture({
            size: [this.context.canvas.width, this.context.canvas.height, 1],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depth_buffer_view = this.depth_buffer.createView();


        this.renderPassDescriptor = {
            colorAttachments: [
                {
                    view: undefined,
                    resolveTarget: undefined,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.12, g: 0.12, b: 0.13, a: 1.0 }
                }
            ],
            depthStencilAttachment: {
                view: this.depth_buffer_view,
                depthLoadOp: "clear",
                depthStoreOp: "store",
                depthClearValue: 1.0,
            },
        }


        const updateButton = document.getElementById("updateValues");
        updateButton.addEventListener("click", this.updateValues.bind(this));
        this.updateValues();

        this.runComputePass();
        this.render();
    }

    private setupComputePipelines(): void {
        // === Workgroup Limits ===
        const limits: WorkgroupLimits = {
            maxTotalThreads: this.device.limits.maxComputeInvocationsPerWorkgroup,
            maxSizeX: this.device.limits.maxComputeWorkgroupSizeX,
            maxSizeY: this.device.limits.maxComputeWorkgroupSizeY,
            maxSizeZ: this.device.limits.maxComputeWorkgroupSizeZ,
        };

        // === Pipeline Layouts ===
        const pipelineLayoutFind = this.device.createPipelineLayout({
            label: 'pipeline-layout-find',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["find"]),
        });

        const pipelineLayoutSort = this.device.createPipelineLayout({
            label: 'pipeline-layout-sort',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["sort"]),
        });

        // === Workgroup Strategy: 2D tiling for rays ===
        const rayGridSize: [number, number, number] = [this.raySamples[0], this.raySamples[1], 1];
        const tile2DGridPerRay: WorkgroupStrategy = ({ totalThreads, problemSize }) => {
            const maxX = Math.floor(Math.sqrt(totalThreads));
            const x = Math.min(problemSize[0], maxX);
            const y = Math.min(problemSize[1], Math.floor(totalThreads / x));

            return {
                workgroupSize: [x, y, 1],
                // No need to override dispatchSize — let it default to problemSize / workgroupSize
            };
        };
        this.workgroupLayoutGrid = this.computeWorkgroupLayout(rayGridSize, limits, tile2DGridPerRay);

        // === Pipeline: findLeaves ===
        const stackSize = 2 * this.scene.tree.depth + 1;

        this.pipelineFindLeaves = this.device.createComputePipeline({
            layout: pipelineLayoutFind,
            compute: {
                module: this.device.createShaderModule({
                    code: this.applyShaderConstants(find_leaves_src, {
                        WORKGROUP_SIZE_X: this.workgroupLayoutGrid.workgroupSize[0],
                        WORKGROUP_SIZE_Y: this.workgroupLayoutGrid.workgroupSize[1],
                        WORKGROUP_SIZE_Z: this.workgroupLayoutGrid.workgroupSize[2],
                        MAX_STACK_SIZE: stackSize,
                    }),
                }),
                entryPoint: "main",
                constants: {
                    TREE_DEPTH: this.scene.tree.depth,
                    BLOCK_SIZE: 128,
                },
            },
        });

        // === Workgroup Strategy: 1 ray per workgroup for sorting ===
        const oneWorkgroupPerRay = (blockSize: number): WorkgroupStrategy =>
            ({ problemSize, totalThreads }) => {
                if (blockSize > totalThreads) throw new Error(`BLOCK_SIZE too large`);
                const rayCount = problemSize[0];
                return {
                    workgroupSize: [blockSize, 1, 1],
                    dispatchSize: [rayCount, 1, 1],
                };
            };

        const rayCount = this.raySamples[0] * this.raySamples[1];
        const sortProblemSize: [number, number, number] = [rayCount, 1, 1];
        this.workgroupLayoutSort = this.computeWorkgroupLayout(sortProblemSize, limits, oneWorkgroupPerRay(128));

        // === Pipeline: bitonicSort ===
        this.pipelineBitonicSort = this.device.createComputePipeline({
            layout: pipelineLayoutSort,
            compute: {
                module: this.device.createShaderModule({
                    code: this.applyShaderConstants(bitonic_sort_src, {
                        WORKGROUP_SIZE: this.workgroupLayoutSort.workgroupSize[0],
                    }),
                }),
                entryPoint: "main",
                constants: {
                    BLOCK_SIZE: 128,
                },
            },
        });
    }

    async createDevice() {
        this.adapter = await navigator.gpu?.requestAdapter();
        if (!this.adapter) {
            console.error("WebGPU: No GPU adapter found!");
            return null;
        }

        const requiredLimits = {
            maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
            maxStorageBuffersPerShaderStage: this.adapter.limits.maxStorageBuffersPerShaderStage
        };


        this.device = await this.adapter.requestDevice({ requiredLimits });

        // Listen for device loss
        this.device.lost.then((info) => {
            console.warn(`WebGPU Device Lost: ${info.message}`);
            this.handleDeviceLost();
        });

        return this.device;
    }

    async handleDeviceLost() {
        console.log("Attempting to recover WebGPU device...");

        // Wait a moment before retrying (avoids potential GPU crashes)
        await new Promise(resolve => setTimeout(resolve, 100));

        // Reinitialize everything
        await this.createDevice();
        if (!this.device) {
            console.error("Failed to recover WebGPU device.");
            return;
        }

        console.log("WebGPU device restored.");
        // Recreate GPU resources here (buffers, pipelines, etc.)
    }

    render = () => {
        let currTime = performance.now() * 0.001;

        const deltaTime = currTime - this.prevTime;
        this.prevTime = currTime;

        this.timeAccumulator += deltaTime;

        while (this.timeAccumulator >= this.timeStep) {
            this.timeAccumulator -= this.timeStep;
        }

        this.calculateFPS(currTime);

        this.runRenderPass();

        requestAnimationFrame(this.render);
    }

    async runComputePass() {
        this.bufferManager.clear("ray_nodes");
        this.bufferManager.clear("node_visibility");


        const computeEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const computePass: GPUComputePassEncoder = computeEncoder.beginComputePass();

        this.computeFindLeaves(computePass);

        const sortNodesCheckbox = <HTMLInputElement>document.getElementById("sortNodes");
        const sortNodes = sortNodesCheckbox.checked;

        if (sortNodes) {
            this.computebitonicSort(computePass);
        }

        computePass.end();
        this.device.queue.submit([computeEncoder.finish()]);
    }

    computeFindLeaves(encoder: GPUComputePassEncoder) {
        const dispatchX = this.workgroupLayoutGrid.dispatchSize[0];
        const dispatchY = this.workgroupLayoutGrid.dispatchSize[1];
        const dispatchZ = this.workgroupLayoutGrid.dispatchSize[2];

        encoder.setPipeline(this.pipelineFindLeaves);
        encoder.setBindGroup(0, this.bindGroupsManager.getGroup("find"));
        encoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }

    computebitonicSort(encoder: GPUComputePassEncoder) {
        const totalWorkgroups = this.workgroupLayoutSort.dispatchSize[0];

        encoder.setPipeline(this.pipelineBitonicSort);
        encoder.setBindGroup(0, this.bindGroupsManager.getGroup("sort"));
        encoder.dispatchWorkgroups(totalWorkgroups);
    }

    runRenderPass() {
        this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 64, new Float32Array(this.scene.camera.viewMatrix));
        this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 128, new Float32Array(this.scene.camera.projectionMatrix));

        const colorTexture = this.context.getCurrentTexture();
        this.renderPassDescriptor.colorAttachments[0].view = colorTexture.createView();

        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

        // Render points
        const renderPointsCheckbox = <HTMLInputElement>document.getElementById("renderPoints");
        const renderPoints = renderPointsCheckbox.checked;
        if (renderPoints) {
            renderPass.setPipeline(this.pipelineRenderPoints);
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            renderPass.setBindGroup(1, this.bindGroupsManager.getGroup("points"));
            renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
            renderPass.setVertexBuffer(1, this.bufferManager.get("colors"));
            const pointsToDraw = this.bufferManager.get("points").size / 16;
            renderPass.draw(pointsToDraw, 1);
        } else {
            renderPass.setPipeline(this.pipelineRenderWireframe);
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            renderPass.draw(4, this.scene.triangleCount); // 1 -> 2 -> 3 -> 1
        }

        // Render rays
        renderPass.setPipeline(this.pipelineRenderRays);
        renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
        renderPass.draw(2 * this.raySamples[0] * this.raySamples[1], 1);

        // Render nodes
        const showNodesCheckbox = <HTMLInputElement>document.getElementById("showNodes");
        const showNodes = showNodesCheckbox.checked;
        if (showNodes) {
            renderPass.setPipeline(this.pipelineRenderNodes);
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("nodes"));
            renderPass.draw(24, Math.pow(4, this.scene.tree.depth));
        }

        renderPass.end();

        const gizmoPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: colorTexture.createView(),
                    loadOp: "load",
                    storeOp: "store",
                },
            ],
        };

        // Render gizmo
        let vr = mat3.create();// View rotation
        mat3.fromMat4(vr, this.scene.camera.viewMatrix);
        let rotationMatrix = mat4.fromValues(
            vr[0], vr[1], vr[2], 0,
            vr[3], vr[4], vr[5], 0,
            vr[6], vr[7], vr[8], 0,
            0, 0, 0, 1);

        let gizmoModel = mat4.create();
        mat4.scale(gizmoModel, gizmoModel, vec3.fromValues(0.1, 0.1, 0.1));
        mat4.multiply(gizmoModel, gizmoModel, rotationMatrix);
        let aspectRatio = this.scene.camera.aspect;
        gizmoModel[12] = aspectRatio - 0.15;
        gizmoModel[13] = 0.85;

        let gizmoView = mat4.create();
        mat4.identity(gizmoView);

        let gizmoProjection = mat4.create();
        mat4.ortho(gizmoProjection, 0, aspectRatio, 0, 1, -2, 1);

        this.bufferManager.write("gizmo_uniforms", new Float32Array(gizmoModel), 0);
        this.bufferManager.write("gizmo_uniforms", new Float32Array(gizmoView), 64);
        this.bufferManager.write("gizmo_uniforms", new Float32Array(gizmoProjection), 128);

        const gizmoPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(gizmoPassDescriptor);
        gizmoPass.setPipeline(this.pipelineRenderGizmo);
        gizmoPass.setBindGroup(0, this.bindGroupsManager.getGroup("gizmo"));
        gizmoPass.setVertexBuffer(0, this.bufferManager.get("gizmo_vertices"));
        gizmoPass.draw(6);
        gizmoPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    updateValues() {
        let oldSamples = [this.raySamples[0], this.raySamples[1]];
        this.raySamples[0] = parseFloat((<HTMLInputElement>document.getElementById("samplesX")).value);
        this.raySamples[1] = parseFloat((<HTMLInputElement>document.getElementById("samplesY")).value);
        const originX = parseFloat((<HTMLInputElement>document.getElementById("originX")).value);
        const originY = parseFloat((<HTMLInputElement>document.getElementById("originY")).value);
        const originZ = parseFloat((<HTMLInputElement>document.getElementById("originZ")).value);
        const startTheta = parseFloat((<HTMLInputElement>document.getElementById("startTheta")).value);
        const endTheta = parseFloat((<HTMLInputElement>document.getElementById("endTheta")).value);
        const startPhi = parseFloat((<HTMLInputElement>document.getElementById("startPhi")).value);
        const endPhi = parseFloat((<HTMLInputElement>document.getElementById("endPhi")).value);
        const renderMode = parseFloat((<HTMLInputElement>document.getElementById("renderMode")).value);

        this.device.queue.writeBuffer(this.bufferManager.get("comp_uniforms"), 0, new Float32Array([
            originX, originY, originZ, // 12
            startTheta, endTheta, startPhi, endPhi // 32
        ]));
        this.device.queue.writeBuffer(this.bufferManager.get("comp_uniforms"), 32, new Uint32Array([
            this.raySamples[0], this.raySamples[1],  // 40
        ]));

        this.bufferManager.write("render_mode", new Uint32Array([renderMode]));

        if (oldSamples[0] != this.raySamples[0] || oldSamples[1] != this.raySamples[1]) {

            // Recreate buffers
            this.bufferManager.resize("rays", this.raySamples[0] * this.raySamples[1] * 2 * 4 * 4);
            this.bufferManager.resize("ray_nodes", this.raySamples[0] * this.raySamples[1] * 128 * 4);
            this.bufferManager.resize("ray_node_counts", this.raySamples[0] * this.raySamples[1] * 4);

            this.bindGroupsManager.updateGroup("find", [
                { binding: 1, resource: { buffer: this.bufferManager.get("rays") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
            ]);

            this.bindGroupsManager.updateGroup("sort", [
                { binding: 1, resource: { buffer: this.bufferManager.get("rays") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
            ]);

            this.bindGroupsManager.updateGroup("render", [
                { binding: 5, resource: { buffer: this.bufferManager.get("rays") } }
            ]);

            this.setupComputePipelines();
        }
        this.runComputePass();
    }

    applyShaderConstants(shaderSrc: string, constants: Record<string, string | number>): string {
        let result = shaderSrc;
        for (const [key, value] of Object.entries(constants)) {
            const placeholder = new RegExp(`__${key}__`, 'g');
            result = result.replace(placeholder, value.toString());
        }
        return result;
    }

    computeWorkgroupLayout(
        problemSize: [number, number, number],
        limits: WorkgroupLimits,
        strategy: WorkgroupStrategy
    ): WorkgroupLayout {
        const result = strategy({
            totalThreads: limits.maxTotalThreads,
            problemSize,
        });

        const [workDimX, workDimY, workDimZ] = result.workgroupSize;
        const totalThreads = workDimX * workDimY * workDimZ;

        // Validate limits
        if (workDimX > limits.maxSizeX)
            throw new Error(`workDimX ${workDimX} exceeds device limit ${limits.maxSizeX}`);
        if (workDimY > limits.maxSizeY)
            throw new Error(`workDimY ${workDimY} exceeds device limit ${limits.maxSizeY}`);
        if (workDimZ > limits.maxSizeZ)
            throw new Error(`workDimZ ${workDimZ} exceeds device limit ${limits.maxSizeZ}`);
        if (totalThreads > limits.maxTotalThreads)
            throw new Error(`Total threads (${totalThreads}) exceed max allowed (${limits.maxTotalThreads})`);

        const dispatchSize = result.dispatchSize ?? [
            Math.ceil(problemSize[0] / workDimX),
            Math.ceil(problemSize[1] / workDimY),
            Math.ceil(problemSize[2] / workDimZ),
        ];

        return {
            workgroupSize: [workDimX, workDimY, workDimZ],
            dispatchSize,
        };
    }


    calculateFPS(currTime: number) {
        this.frameCount++;

        // Calculate FPS every second
        const elapsedTime = currTime - this.fpsLastTime;
        if (elapsedTime > 1) {
            this.fps = this.frameCount / elapsedTime;
            this.frameCount = 0;
            this.fpsLastTime = currTime;
        }

        const fpsLabel: HTMLElement = <HTMLElement>document.getElementById("fps");
        fpsLabel.innerText = (this.fps).toFixed(2);
    }
}