import generate_rays_src from "./shaders/compute/ray-gen.wgsl"
import find_leaves_src from "./shaders/compute/find-leaves.wgsl"
import bitonic_sort_src from "./shaders/compute/bitonic-sort.wgsl"
import collision_src from "./shaders/compute/collision.wgsl"

import wireframe_src from "./shaders/render/wireframe.wgsl"
import points_src from "./shaders/render/points.wgsl"
import rays_src from "./shaders/render/rays.wgsl"
import gizmo_src from "./shaders/render/gizmo.wgsl"
import nodes_src from "./shaders/render/nodes.wgsl"

import { Scene } from "./Scene";
import { WorkgroupStrategy } from "./types/types"
import { BufferManager } from "./BufferManager"
import { QuadTree } from "./Optimization"
import { BindGroupManager } from "./BindGroupsManager"
import { PipelineManager } from "./PipelineManager"
import { WorkgroupManager } from "./WorkgroupManager"
import { Utils } from "./Utils"


export class Renderer {
    canvas: HTMLCanvasElement

    // Device/Context objects
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    renderPassDescriptor: GPURenderPassDescriptor;

    //Assets
    depthTexture: GPUTexture;
    depthView: GPUTextureView;

    raySamples: Uint32Array = new Uint32Array([1, 1]);

    // Managers
    bufferManager: BufferManager;
    bindGroupsManager: BindGroupManager;
    pipelineManager: PipelineManager;
    workgroups: WorkgroupManager;

    // Scene to render
    scene: Scene

    private canRender = {
        gizmo: true,
        rays: true,
        points: false,
        nodes: false,
        mesh: false,
    };

    // Time
    private prevTime = 0;
    private timeAccumulator = 0;
    private readonly timeStep = 1 / 60;
    private fps = 0;
    private fpsLastTime = 0;
    private frameCount = 0;
    private animationFrameId: number | null = null;
    private running = false;

    constructor(canvas: HTMLCanvasElement, scene: Scene, device: GPUDevice) {
        this.canvas = canvas;
        this.scene = scene;
        this.device = device;

        this.bufferManager = new BufferManager(this.device);
        this.bindGroupsManager = new BindGroupManager(this.device);
        this.pipelineManager = new PipelineManager(this.device);

        this.workgroups = new WorkgroupManager(this.device);
    }

    async init() {
        this.context = <GPUCanvasContext>this.canvas.getContext("webgpu");
        this.format = "bgra8unorm";

        this.configureContext();

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
                size: this.scene.gizmo.vertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                data: this.scene.gizmo.vertices
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
                size: this.raySamples[0] * this.raySamples[1] * QuadTree.noMaxNodesHit(this.scene.tree.depth) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "ray_node_counts",
                size: this.raySamples[0] * this.raySamples[1] * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "debug_distance",
                size: this.raySamples[0] * this.raySamples[1] * QuadTree.noMaxNodesHit(this.scene.tree.depth) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            }
        ]);

        this.bindGroupsManager.createLayout({
            name: "compute-uniforms",
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
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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
            name: "compute-uniforms",
            layoutName: "compute-uniforms",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("comp_uniforms") } },
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

        this.pipelineManager.create({
            name: "render-points",
            type: "render",
            layout: pipeline_layout_points,
            code: points_src,
            render: {
                vertex: {
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
                        },
                        {
                            arrayStride: 4, // 1 uint32 @ 4 bytes
                            attributes: [
                                {
                                    shaderLocation: 2, // Classification
                                    offset: 0,
                                    format: 'uint32'
                                }
                            ]
                        }
                    ]
                },
                fragment: {
                    entryPoint: "main_fs",
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
                primitive: { topology: 'point-list' },
                depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
            }
        });

        this.pipelineManager.create({
            name: "render-wireframe",
            type: "render",
            layout: pipeline_layout_render,
            code: wireframe_src,
            render: {
                vertex: {
                    entryPoint: 'main',
                },
                fragment: {
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
            }
        });

        this.pipelineManager.create({
            name: "render-rays",
            type: "render",
            layout: pipeline_layout_render,
            code: rays_src,
            render: {
                vertex: {
                    entryPoint: 'main_vs',
                },
                fragment: {
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
            }
        });

        this.pipelineManager.create({
            name: "render-gizmo",
            type: "render",
            layout: pipeline_layout_gizmo,
            code: gizmo_src,
            render: {
                vertex: {
                    entryPoint: 'main',
                    buffers: [
                        {
                            arrayStride: 4 * 4,
                            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }],
                        }
                    ],
                },
                fragment: {
                    entryPoint: 'main_fs',
                    targets: [
                        { format: 'bgra8unorm' }
                    ]
                },
                primitive: {
                    topology: 'line-strip',
                }
            }
        });

        this.pipelineManager.create({
            name: "render-nodes",
            type: "render",
            layout: pipeline_layout_nodes,
            code: nodes_src,
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth),
                RAY_COUNT: this.raySamples[0] * this.raySamples[1],
            },
            render: {
                vertex: {
                    entryPoint: 'main',
                    buffers: [
                        {
                            arrayStride: 4 * 4,
                            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }],
                        }
                    ],
                },
                fragment: {
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
                        }
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
            }
        });

        this.depthTexture = this.device.createTexture({
            size: [this.context.canvas.width, this.context.canvas.height, 1],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthView = this.depthTexture.createView();


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
                view: this.depthView,
                depthLoadOp: "clear",
                depthStoreOp: "store",
                depthClearValue: 1.0,
            },
        }

        document.getElementById("raySampleInputs")?.addEventListener("change", this.updateRaySamples.bind(this));
        document.getElementById("originInputs")?.addEventListener("change", this.updateRayOrigin.bind(this));
        document.getElementById("thetaPhiInputs")?.addEventListener("change", this.updateThetaPhi.bind(this));
        document.getElementById("runNodes")?.addEventListener("click", this.runComputePass.bind(this));
        document.getElementById("runPoints")?.addEventListener("click", this.runCollision.bind(this));
        document.getElementById("renderMode")?.addEventListener("change", this.updateRenderMode.bind(this));

        this.updateRaySamples();
        this.updateRayOrigin();
        this.updateThetaPhi();
        this.runComputePass();
        this.updateRenderMode();
    }

    private configureContext() {
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: "opaque",
        });
    }

    private createDepthTexture(width: number, height: number) {
        this.depthTexture = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.depthView = this.depthTexture.createView();
        this.renderPassDescriptor.depthStencilAttachment.view = this.depthView;
    }

    async setPointData() {
        this.bufferManager.resize("points", this.scene.points.byteLength);
        this.bufferManager.write("points", this.scene.points);

        this.bufferManager.resize("colors", this.scene.colors.byteLength);
        this.bufferManager.write("colors", this.scene.colors);

        this.bufferManager.resize("classification", this.scene.classification.byteLength);
        this.bufferManager.write("classification", this.scene.classification);

        this.bufferManager.resize("point_visibility", Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT);

        this.bindGroupsManager.updateGroup("render", [
            { binding: 0, resource: { buffer: this.bufferManager.get("points") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("point_visibility") } },
        ]);

        this.bindGroupsManager.updateGroup("collision", [
            { binding: 0, resource: { buffer: this.bufferManager.get("points") } },
            { binding: 3, resource: { buffer: this.bufferManager.get("point_visibility") } },
        ]);

        this.canRender.points = true;
    }

    async setNodeData(resize: boolean = true) {
        if (resize) {
            const rayCount = this.raySamples[0] * this.raySamples[1];
            const maxNodesHit = QuadTree.noMaxNodesHit(this.scene.tree.depth);

            this.bufferManager.resize("nodes", QuadTree.totalNodes(this.scene.tree.depth) * QuadTree.BYTES_PER_NODE * 4);
            this.bufferManager.resize("point_to_node", this.scene.tree.root.pointCount * 4);
            this.bufferManager.resize("node_visibility", QuadTree.leafNodes(this.scene.tree.depth) / 32 * Uint32Array.BYTES_PER_ELEMENT);
            this.bufferManager.resize("ray_nodes", rayCount * maxNodesHit * 4);
            this.bufferManager.resize("debug_distance", rayCount * maxNodesHit * 4);
            this.bufferManager.resize("copy_distances_buffer", rayCount * maxNodesHit * 4);
            this.bufferManager.resize("copy_indices_buffer", rayCount * maxNodesHit * 4);
        }

        this.bufferManager.write("nodes", this.scene.tree.flatten());
        this.bufferManager.write("point_to_node", this.scene.tree.mapPointsToNodes());

        this.bindGroupsManager.updateGroup("find", [
            { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("ray_nodes") } },
            { binding: 3, resource: { buffer: this.bufferManager.get("node_visibility") } },
        ]);

        this.bindGroupsManager.updateGroup("sort", [
            { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("ray_nodes") } },
            { binding: 3, resource: { buffer: this.bufferManager.get("debug_distance") } },
        ]);

        this.bindGroupsManager.updateGroup("points", [
            { binding: 1, resource: { buffer: this.bufferManager.get("nodes") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("point_to_node") } },
        ]);

        this.bindGroupsManager.updateGroup("nodes", [
            { binding: 1, resource: { buffer: this.bufferManager.get("nodes") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("node_visibility") } },
            { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
        ]);

        this.pipelineManager.update("render-nodes", {
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            }
        });

        this.pipelineManager.update("find-leaves", {
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            },
            codeConstants: {
                MAX_STACK_SIZE: 2 ** this.scene.tree.depth + 1
            }
        });

        this.pipelineManager.update("collision", {
            constants: {
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            }
        });

        this.workgroups.update("workgroup-per-ray", {
            strategyArgs: [QuadTree.noMaxNodesHit(this.scene.tree.depth)],
        });
        const linearWorkgroup = this.workgroups.getLayout("workgroup-per-ray");

        this.pipelineManager.update("bitonic-sort", {
            constants: {
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            },
            codeConstants: {
                WORKGROUP_SIZE: linearWorkgroup.workgroupSize[0],
            }
        });

        this.canRender.nodes = true;
    }

    async setMeshData() {
        this.bufferManager.resize("indices", this.scene.indices.byteLength);
        this.bufferManager.write("indices", this.scene.indices);

        this.bufferManager.resize("node_to_triangle", this.scene.nodeToTriangles.byteLength);
        this.bufferManager.write("node_to_triangle", this.scene.nodeToTriangles);

        this.bindGroupsManager.updateGroup("render", [
            { binding: 1, resource: { buffer: this.bufferManager.get("indices") } },
        ]);

        this.bindGroupsManager.updateGroup("collision", [
            { binding: 1, resource: { buffer: this.bufferManager.get("indices") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("node_to_triangle") } },
        ]);

        this.canRender.mesh = true;
    }

    private setupComputePipelines(): void {
        // === Pipeline Layouts ===

        const pipelineLayoutRays = this.device.createPipelineLayout({
            label: 'pipeline-layout-rays',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["compute-uniforms", "rays"]),
        });

        const pipelineLayoutFind = this.device.createPipelineLayout({
            label: 'pipeline-layout-find',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["compute-uniforms", "rays", "find"]),
        });

        const pipelineLayoutSort = this.device.createPipelineLayout({
            label: 'pipeline-layout-sort',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["compute-uniforms", "rays", "sort"]),
        });

        const pipelineLayoutCollision = this.device.createPipelineLayout({
            label: 'pipeline-layout-collision',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["compute-uniforms", "rays", "sort", "collision"]),
        });


        // === Workgroup Strategy: 2D tiling for rays ===
        const tile2DGridPerRay = (): WorkgroupStrategy =>
            ({ problemSize, limits }) => {
                const maxX = Math.floor(Math.sqrt(limits.maxTotalThreads));
                const x = Math.min(problemSize[0], maxX);
                const y = Math.min(problemSize[1], Math.floor(limits.maxTotalThreads / x));

                return {
                    workgroupSize: [x, y, 1],
                    // No need to override dispatchSize — let it default to problemSize / workgroupSize
                };
            };

        this.workgroups.register({
            name: "2d-grid-per-ray",
            problemSize: [this.raySamples[0], this.raySamples[1], 1],
            strategyFn: tile2DGridPerRay,
            strategyArgs: [],
        });

        // === Pipeline: findLeaves ===
        const stackSize = 2 * this.scene.tree.depth + 1;
        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");

        this.pipelineManager.create({
            name: "generate-rays",
            type: "compute",
            layout: pipelineLayoutRays,
            code: generate_rays_src,
            codeConstants: {
                WORKGROUP_SIZE_X: gridLayout.workgroupSize[0],
                WORKGROUP_SIZE_Y: gridLayout.workgroupSize[1],
                WORKGROUP_SIZE_Z: gridLayout.workgroupSize[2],
            },
            compute: {
                entryPoint: "main",
            }
        });

        this.pipelineManager.create({
            name: "find-leaves",
            type: "compute",
            layout: pipelineLayoutFind,
            code: find_leaves_src,
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth),
            },
            codeConstants: {
                WORKGROUP_SIZE_X: gridLayout.workgroupSize[0],
                WORKGROUP_SIZE_Y: gridLayout.workgroupSize[1],
                WORKGROUP_SIZE_Z: gridLayout.workgroupSize[2],
                MAX_STACK_SIZE: stackSize,
            },
            compute: {
                entryPoint: "main",
            }
        });


        this.pipelineManager.create({
            name: "collision",
            type: "compute",
            layout: pipelineLayoutCollision,
            code: collision_src,
            constants: {
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth),
            },
            codeConstants: {
                WORKGROUP_SIZE_X: gridLayout.workgroupSize[0],
                WORKGROUP_SIZE_Y: gridLayout.workgroupSize[1],
                WORKGROUP_SIZE_Z: gridLayout.workgroupSize[2],
            },
            compute: {
                entryPoint: "main",
            }
        });

        // === Workgroup Strategy: 1 ray per workgroup for sorting ===
        // TODO: This should go up to maximum workgroup size if we have too many rays. Going to have to also change to shader to handle this.
        const multiRayPerWorkgroup = (blockSize: number): WorkgroupStrategy =>
            ({ problemSize, limits }) => {
                const totalRays = problemSize[0];
                const maxThreads = limits.maxTotalThreads;

                if (blockSize > maxThreads) {
                    throw new Error(`BLOCK_SIZE (${blockSize}) exceeds max workgroup size (${maxThreads})`);
                }

                const maxBlocksPerGroup = Math.floor(maxThreads / blockSize);
                const totalBlocks = totalRays;

                const blocksPerGroup = Math.min(maxBlocksPerGroup, totalBlocks);

                if (blocksPerGroup === 0) {
                    throw new Error(`BLOCK_SIZE too large to fit even one block in a workgroup`);
                }

                const dispatchSizeX = Math.ceil(totalBlocks / blocksPerGroup);

                if (dispatchSizeX > limits.maxDispatch) {
                    throw new Error(`Dispatch size (${dispatchSizeX}) exceeds device limit (${limits.maxDispatch})`);
                }

                return {
                    workgroupSize: [blocksPerGroup * blockSize, 1, 1],
                    dispatchSize: [dispatchSizeX, 1, 1],
                };
            };

        this.workgroups.register({
            name: "workgroup-per-ray",
            problemSize: [this.raySamples[0] * this.raySamples[1], 1, 1],
            strategyFn: multiRayPerWorkgroup,
            strategyArgs: [QuadTree.noMaxNodesHit(this.scene.tree.depth)],
        });

        const linearLayout = this.workgroups.getLayout("workgroup-per-ray");

        // === Pipeline: bitonicSort ===
        this.pipelineManager.create({
            name: "bitonic-sort",
            type: "compute",
            layout: pipelineLayoutSort,
            code: bitonic_sort_src,
            constants: {
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth),
            },
            codeConstants: {
                WORKGROUP_SIZE: linearLayout.workgroupSize[0],
            },
            compute: {
                entryPoint: "main",
            }
        });
    }

    startRendering() {
        if (!this.running) {
            this.running = true;
            this.prevTime = performance.now() * 0.001;
            this.render();
        }
    }

    stopRendering() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        this.running = false;
    }

    render = () => {
        if (!this.running) return;

        const currTime = performance.now() * 0.001;
        const deltaTime = currTime - this.prevTime;
        this.prevTime = currTime;
        this.timeAccumulator += deltaTime;

        while (this.timeAccumulator >= this.timeStep) {
            this.timeAccumulator -= this.timeStep;
        }

        this.calculateFPS(currTime);
        this.runRenderPass();

        this.animationFrameId = requestAnimationFrame(this.render);
    };

    async runComputePass() {
        this.bufferManager.clear("ray_nodes");
        this.bufferManager.clear("debug_distance");
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

        let distancesBuffer: GPUBuffer | undefined;
        let indicesBuffer: GPUBuffer | undefined;

        const rayCount = this.raySamples[0] * this.raySamples[1];
        const bufferSize = rayCount * QuadTree.noMaxNodesHit(this.scene.tree.depth) * 4;

        distancesBuffer = this.bufferManager.get("copy_distances_buffer") ?? this.bufferManager.createBuffer(
            "copy_distances_buffer",
            bufferSize,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        );

        indicesBuffer = this.bufferManager.get("copy_indices_buffer") ?? this.bufferManager.createBuffer(
            "copy_indices_buffer",
            bufferSize,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        );

        computeEncoder.copyBufferToBuffer(
            this.bufferManager.get("debug_distance"), 0,
            distancesBuffer, 0,
            bufferSize
        );

        computeEncoder.copyBufferToBuffer(
            this.bufferManager.get("ray_nodes"), 0,
            indicesBuffer, 0,
            bufferSize
        );

        this.device.queue.submit([computeEncoder.finish()]);

        if (distancesBuffer && indicesBuffer) {
            // Map both buffers in parallel
            await Promise.all([
                distancesBuffer.mapAsync(GPUMapMode.READ),
                indicesBuffer.mapAsync(GPUMapMode.READ),
            ]);

            const distanceArray = new Float32Array(distancesBuffer.getMappedRange().slice(0));
            const indexArray = new Uint32Array(indicesBuffer.getMappedRange().slice(0));

            const rayCount = this.raySamples[0] * this.raySamples[1];
            const blockSize = QuadTree.noMaxNodesHit(this.scene.tree.depth);
            Utils.displayRayData(indexArray, distanceArray, rayCount, blockSize);
        }

        distancesBuffer.unmap();
        indicesBuffer.unmap();
    }

    runGenerateRays() {
        this.bufferManager.clear("rays");

        const computeEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const computePass: GPUComputePassEncoder = computeEncoder.beginComputePass();

        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");
        const dispatchX = gridLayout.dispatchSize[0];
        const dispatchY = gridLayout.dispatchSize[1];
        const dispatchZ = gridLayout.dispatchSize[2];

        computePass.setPipeline(this.pipelineManager.get<GPUComputePipeline>("generate-rays"));
        computePass.setBindGroup(0, this.bindGroupsManager.getGroup("compute-uniforms"));
        computePass.setBindGroup(1, this.bindGroupsManager.getGroup("rays"));
        computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);

        computePass.end();
        this.device.queue.submit([computeEncoder.finish()]);
    }

    async runCollision() {
        this.bufferManager.clear("point_visibility");

        const computeEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const computePass: GPUComputePassEncoder = computeEncoder.beginComputePass();

        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");
        const dispatchX = gridLayout.dispatchSize[0];
        const dispatchY = gridLayout.dispatchSize[1];
        const dispatchZ = gridLayout.dispatchSize[2];

        computePass.setPipeline(this.pipelineManager.get<GPUComputePipeline>("collision"));
        computePass.setBindGroup(0, this.bindGroupsManager.getGroup("compute-uniforms"));
        computePass.setBindGroup(1, this.bindGroupsManager.getGroup("rays"));
        computePass.setBindGroup(2, this.bindGroupsManager.getGroup("sort"));
        computePass.setBindGroup(3, this.bindGroupsManager.getGroup("collision"));
        computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);

        computePass.end();

        const pointVisibilityBuffer = this.bufferManager.get("point_visibility");
        const visibilityByteLength = pointVisibilityBuffer.size;
        const readBuffer = this.device.createBuffer({
            size: visibilityByteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        computeEncoder.copyBufferToBuffer(
            pointVisibilityBuffer, 0,
            readBuffer, 0,
            visibilityByteLength
        );

        this.device.queue.submit([computeEncoder.finish()]);

        await this.device.queue.onSubmittedWorkDone();
        await readBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readBuffer.getMappedRange();
        const visibilityData = new Uint32Array(arrayBuffer.slice(0));

        const visiblePoints: number[] = [];

        for (let wordIndex = 0; wordIndex < visibilityData.length; wordIndex++) {
            const word = visibilityData[wordIndex];
            if (word === 0) continue; // skip empty words

            for (let bit = 0; bit < 32; bit++) {
                if ((word & (1 << bit)) !== 0) {
                    visiblePoints.push(wordIndex * 32 + bit);
                }
            }
        }

        readBuffer.unmap();

        console.log(`Visible point indices (${visiblePoints.length}):`, visiblePoints);
    }

    computeFindLeaves(pass: GPUComputePassEncoder) {
        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");
        const dispatchX = gridLayout.dispatchSize[0];
        const dispatchY = gridLayout.dispatchSize[1];
        const dispatchZ = gridLayout.dispatchSize[2];

        pass.setPipeline(this.pipelineManager.get<GPUComputePipeline>("find-leaves"));
        pass.setBindGroup(0, this.bindGroupsManager.getGroup("compute-uniforms"));
        pass.setBindGroup(1, this.bindGroupsManager.getGroup("rays"));
        pass.setBindGroup(2, this.bindGroupsManager.getGroup("find"));
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }

    computebitonicSort(pass: GPUComputePassEncoder) {
        const linearLayout = this.workgroups.getLayout("workgroup-per-ray");
        const totalWorkgroups = linearLayout.dispatchSize[0];

        pass.setPipeline(this.pipelineManager.get<GPUComputePipeline>("bitonic-sort"));
        pass.setBindGroup(0, this.bindGroupsManager.getGroup("compute-uniforms"));
        pass.setBindGroup(1, this.bindGroupsManager.getGroup("rays"));
        pass.setBindGroup(2, this.bindGroupsManager.getGroup("sort"));
        pass.dispatchWorkgroups(totalWorkgroups);
    }

    runRenderPass() {
        this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 64, new Float32Array(this.scene.camera.viewMatrix));
        this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 128, new Float32Array(this.scene.camera.projectionMatrix));

        const colorTexture = this.context.getCurrentTexture();
        this.renderPassDescriptor.colorAttachments[0].view = colorTexture.createView();

        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

        const renderPointsCheckbox = <HTMLInputElement>document.getElementById("renderPoints");
        const renderPoints = renderPointsCheckbox.checked;
        const renderMeshCheckbox = <HTMLInputElement>document.getElementById("renderMesh");
        const renderMesh = renderMeshCheckbox.checked;
        if (this.canRender.points && renderPoints) {
            // Render points
            renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-points"));
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            renderPass.setBindGroup(1, this.bindGroupsManager.getGroup("points"));
            renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
            renderPass.setVertexBuffer(1, this.bufferManager.get("colors"));
            renderPass.setVertexBuffer(2, this.bufferManager.get("classification"));
            const pointsToDraw = this.bufferManager.get("points").size / 16;
            renderPass.draw(pointsToDraw, 1);
        } else if (this.canRender.mesh && renderMesh) {
            // Render wireframe
            renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-wireframe"));
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
            renderPass.draw(4, this.scene.triangleCount); // 1 -> 2 -> 3 -> 1
        }

        // Render rays
        const renderRaysCheckbox = <HTMLInputElement>document.getElementById("renderRays");
        const renderRays = renderRaysCheckbox.checked;
        if (this.canRender.rays && renderRays) {
            renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-rays"));
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            renderPass.draw(2 * this.raySamples[0] * this.raySamples[1], 1);
        }

        // Render nodes
        const showNodesCheckbox = <HTMLInputElement>document.getElementById("showNodes");
        const showNodes = showNodesCheckbox.checked;
        if (this.canRender.nodes && showNodes) {
            renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-nodes"));
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("nodes"));
            renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
            renderPass.draw(24, QuadTree.leafNodes(this.scene.tree.depth));
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

        const { gmodel, gview, gprojection } = this.scene.gizmo.getModelViewProjection(
            this.scene.camera, this.canvas.width, this.canvas.height);

        this.bufferManager.write("gizmo_uniforms", gmodel, 0);
        this.bufferManager.write("gizmo_uniforms", gview, 64);
        this.bufferManager.write("gizmo_uniforms", gprojection, 128);

        const gizmoPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(gizmoPassDescriptor);
        gizmoPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-gizmo"));
        gizmoPass.setBindGroup(0, this.bindGroupsManager.getGroup("gizmo"));
        gizmoPass.setVertexBuffer(0, this.bufferManager.get("gizmo_vertices"));
        gizmoPass.draw(6);
        gizmoPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    updateRaySamples() {
        const [oldX, oldY] = this.raySamples;
        this.raySamples[0] = parseInt((<HTMLInputElement>document.getElementById("samplesX")).value);
        this.raySamples[1] = parseInt((<HTMLInputElement>document.getElementById("samplesY")).value);

        this.device.queue.writeBuffer(this.bufferManager.get("comp_uniforms"), 32, new Uint32Array([
            this.raySamples[0], this.raySamples[1],
        ]));

        if (oldX !== this.raySamples[0] || oldY !== this.raySamples[1]) {
            this.resizeRayRelatedBuffers();
            this.updateRayRelatedBindGroups();
            this.updateRayWorkgroupsAndPipelines();
        }
        this.runGenerateRays();
    }

    updateThetaPhi() {
        const startTheta = parseFloat((<HTMLInputElement>document.getElementById("startTheta")).value);
        const endTheta = parseFloat((<HTMLInputElement>document.getElementById("endTheta")).value);
        const startPhi = parseFloat((<HTMLInputElement>document.getElementById("startPhi")).value);
        const endPhi = parseFloat((<HTMLInputElement>document.getElementById("endPhi")).value);
        this.device.queue.writeBuffer(this.bufferManager.get("comp_uniforms"), 12, new Float32Array([startTheta, endTheta, startPhi, endPhi]));
        this.runGenerateRays();
    }

    updateRayOrigin() {
        const ox = parseFloat((<HTMLInputElement>document.getElementById("originX")).value);
        const oy = parseFloat((<HTMLInputElement>document.getElementById("originY")).value);
        const oz = parseFloat((<HTMLInputElement>document.getElementById("originZ")).value);
        this.device.queue.writeBuffer(this.bufferManager.get("comp_uniforms"), 0, new Float32Array([ox, oy, oz]));
        this.runGenerateRays();
    }

    updateRenderMode() {
        const renderMode = parseInt((<HTMLInputElement>document.getElementById("renderMode")).value);
        this.bufferManager.write("render_mode", new Uint32Array([renderMode]));
    }

    resizeRayRelatedBuffers() {
        const rayCount = this.raySamples[0] * this.raySamples[1];
        const maxNodesHit = QuadTree.noMaxNodesHit(this.scene.tree.depth);

        this.bufferManager.resize("rays", rayCount * 2 * 4 * 4);
        this.bufferManager.resize("ray_nodes", rayCount * maxNodesHit * 4);
        this.bufferManager.resize("debug_distance", rayCount * maxNodesHit * 4);
        this.bufferManager.resize("ray_node_counts", rayCount * 4);
        this.bufferManager.resize("ray_node_counts", rayCount * 4);
        this.bufferManager.resize("copy_distances_buffer", rayCount * maxNodesHit * 4);
        this.bufferManager.resize("copy_indices_buffer", rayCount * maxNodesHit * 4);
    }

    updateRayRelatedBindGroups() {
        this.bindGroupsManager.updateGroup("rays", [
            { binding: 0, resource: { buffer: this.bufferManager.get("rays") } },
        ]);

        this.bindGroupsManager.updateGroup("find", [
            { binding: 1, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("ray_nodes") } },
        ]);

        this.bindGroupsManager.updateGroup("sort", [
            { binding: 1, resource: { buffer: this.bufferManager.get("ray_node_counts") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("ray_nodes") } },
            { binding: 3, resource: { buffer: this.bufferManager.get("debug_distance") } },
        ]);

        this.bindGroupsManager.updateGroup("render", [
            { binding: 5, resource: { buffer: this.bufferManager.get("rays") } }
        ]);

        this.bindGroupsManager.updateGroup("nodes", [
            { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
        ]);
    }

    updateRayWorkgroupsAndPipelines() {
        this.workgroups.update("2d-grid-per-ray", {
            problemSize: [this.raySamples[0], this.raySamples[1], 1],
        });
        this.workgroups.update("workgroup-per-ray", {
            problemSize: [this.raySamples[0] * this.raySamples[1], 1, 1],
        });
        const linearLayout = this.workgroups.getLayout("workgroup-per-ray");
        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");

        this.pipelineManager.update("generate-rays", {
            codeConstants: {
                WORKGROUP_SIZE_X: gridLayout.workgroupSize[0],
                WORKGROUP_SIZE_Y: gridLayout.workgroupSize[1],
                WORKGROUP_SIZE_Z: gridLayout.workgroupSize[2],
            }
        });

        this.pipelineManager.update("find-leaves", {
            codeConstants: {
                WORKGROUP_SIZE_X: gridLayout.workgroupSize[0],
                WORKGROUP_SIZE_Y: gridLayout.workgroupSize[1],
                WORKGROUP_SIZE_Z: gridLayout.workgroupSize[2],
            }
        });

        this.pipelineManager.update("collision", {
            codeConstants: {
                WORKGROUP_SIZE_X: gridLayout.workgroupSize[0],
                WORKGROUP_SIZE_Y: gridLayout.workgroupSize[1],
                WORKGROUP_SIZE_Z: gridLayout.workgroupSize[2],
            }
        });

        this.pipelineManager.update("bitonic-sort", {
            codeConstants: {
                WORKGROUP_SIZE: linearLayout.workgroupSize[0],
            },
        });

        this.pipelineManager.update("render-nodes", {
            constants: {
                RAY_COUNT: this.raySamples[0] * this.raySamples[1],
            }
        });
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

    reset() {
        this.canRender = {
            gizmo: true,
            rays: true,
            points: false,
            mesh: false,
            nodes: false,
        }
        this.stopRendering();
        this.init();
        this.startRendering();
    }

    resize(width: number, height: number) {
        this.canvas.width = width;
        this.canvas.height = height;

        this.configureContext();

        this.createDepthTexture(width, height);
    }
}