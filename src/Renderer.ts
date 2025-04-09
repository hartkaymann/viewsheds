import generate_rays_src from "./shaders/compute/ray-gen.wgsl"
import find_leaves_src from "./shaders/compute/find-leaves.wgsl"
import bitonic_sort_src from "./shaders/compute/bitonic-sort.wgsl"
import collision_src from "./shaders/compute/collision.wgsl"

import wireframe_src from "./shaders/render/wireframe.wgsl"
import points_src from "./shaders/render/points.wgsl"
import rays_src from "./shaders/render/rays.wgsl"
import gizmo_src from "./shaders/render/gizmo.wgsl"
import nodes_src from "./shaders/render/nodes.wgsl"
import composite_src from "./shaders/render/composite.wgsl"

import { Scene } from "./Scene";
import { WorkgroupStrategy } from "./types/types"
import { BufferManager } from "./BufferManager"
import { QuadTree } from "./Optimization"
import { BindGroupManager } from "./BindGroupsManager"
import { PipelineManager } from "./PipelineManager"
import { WorkgroupManager } from "./WorkgroupManager"
import { Utils } from "./Utils"
import { Profiler } from "./Profiler"
import { UIController } from "./ui/UIController "


export class Renderer {
    canvas: HTMLCanvasElement
    ui: UIController | null = null;

    // Device/Context objects
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    renderPassDescriptor: GPURenderPassDescriptor;
    transparentPassDescriptor: GPURenderPassDescriptor;
    compositePassDescriptor: GPURenderPassDescriptor;

    //Assets
    depthTexture: GPUTexture;
    accumTexture: GPUTexture;
    revealageTexture: GPUTexture;
    depthView: GPUTextureView;

    raySamples: [number, number] = [1, 1];

    // Managers
    bufferManager: BufferManager;
    bindGroupsManager: BindGroupManager;
    pipelineManager: PipelineManager;
    workgroups: WorkgroupManager;

    profiler: Profiler;

    // Scene to render
    scene: Scene

    // Settings
    private canRender = {
        gizmo: true,
        rays: true,
        points: false,
        nodes: false,
        mesh: false,
    };

    renderPoints: boolean = true;
    renderMesh: boolean = false;
    renderRays: boolean = true;
    renderNodes: boolean = true;
    runningPanorama: boolean = false;

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
        this.bindGroupsManager = new BindGroupManager(this.device, this.bufferManager);
        this.pipelineManager = new PipelineManager(this.device);
        this.workgroups = new WorkgroupManager(this.device);

        this.profiler = new Profiler(this.device);
        this.profiler.setBufferManager(this.bufferManager);
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
            },
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

        this.bindGroupsManager.createLayout({
            name: "composite",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            ]
        });

        this.createDepthTexture(this.context.canvas.width, this.context.canvas.height);
        this.createOITTextures(this.context.canvas.width, this.context.canvas.height);

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

        this.bindGroupsManager.createGroup({
            name: "composite",
            layoutName: "composite",
            entries: [
                {
                    binding: 0, resource: this.device.createSampler({
                        magFilter: "linear",
                        minFilter: "linear"
                    }),
                },
                { binding: 1, resource: this.accumTexture.createView() },
                { binding: 2, resource: this.revealageTexture.createView() },
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
        const pipeline_layout_composite = this.device.createPipelineLayout({
            label: 'pipeline-layout-composite',
            bindGroupLayouts: this.bindGroupsManager.getLayouts(["composite"])
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

        this.pipelineManager.create({
            name: "transparent",
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
                    entryPoint: 'main_fs',
                    targets: [{
                        format: 'rgba16float',
                        blend: {
                            color: {
                                srcFactor: "one",
                                dstFactor: "one",
                                operation: "add",
                            },
                            alpha: {
                                srcFactor: "zero",
                                dstFactor: "one",
                                operation: "add",
                            },
                        },
                    },
                    {
                        format: 'r16float',
                        blend: {
                            color: {
                                srcFactor: "zero",
                                dstFactor: "one-minus-src-alpha",
                                operation: "add",
                            },
                            alpha: {
                                srcFactor: "zero",
                                dstFactor: "one",
                                operation: "add",
                            },
                        },
                    },
                    ]
                },
                primitive: { topology: 'point-list' },
                depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
            },
        });

        this.pipelineManager.create({
            name: "composite",
            type: "render",
            layout: pipeline_layout_composite,
            code: composite_src,
            render: {
                vertex: {
                    entryPoint: 'main'
                },
                fragment: {
                    entryPoint: 'main_fs',
                    targets: [{ format: 'bgra8unorm' }]
                },
                primitive: {
                    topology: "triangle-list",
                    cullMode: "none",
                },
            },
        });

        this.renderPassDescriptor = {
            label: 'pass-render',
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
        };

        this.transparentPassDescriptor = {
            label: 'pass-transparent',
            colorAttachments: [
                {
                    view: this.accumTexture.createView(),
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                },
                {
                    view: this.revealageTexture.createView(),
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 0 }, // Only R is used
                },
            ],
            depthStencilAttachment: {
                view: this.depthView,
                depthLoadOp: "clear",
                depthStoreOp: "store",
                depthClearValue: 1.0,
            },
        };

        this.compositePassDescriptor = {
            colorAttachments: [
                {
                    view: undefined,
                    loadOp: "clear",
                    storeOp: "store",
                    clearValue: { r: 0.12, g: 0.12, b: 0.13, a: 1.0 },
                }
            ],
        };

        this.updateRaySamples();
        this.updateRayOrigin();
        this.updateThetaPhi();
        this.runGenerateRays();
        this.runNodeCollision();
        this.updateRenderMode();
    }

    private configureContext() {
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: "opaque",
        });
    }

    private createOITTextures(width: number, height: number) {
        this.accumTexture = this.device.createTexture({
            size: [width, height],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        this.revealageTexture = this.device.createTexture({
            size: [width, height],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        if (!this.bindGroupsManager.getGroup("composite"))
            return;

        this.bindGroupsManager.updateGroup("composite", [
            { binding: 1, resource: this.accumTexture.createView() },
            { binding: 2, resource: this.revealageTexture.createView() },
        ]);
    }

    private createDepthTexture(width: number, height: number) {
        this.depthTexture = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthView = this.depthTexture.createView();
    }

    async setPointData() {
        this.bufferManager.resize("points", this.scene.points.byteLength);
        this.bufferManager.write("points", this.scene.points);

        this.bufferManager.resize("colors", this.scene.colors.byteLength);
        this.bufferManager.write("colors", this.scene.colors);

        this.bufferManager.resize("classification", this.scene.classification.byteLength);
        this.bufferManager.write("classification", this.scene.classification);

        this.bufferManager.resize("point_visibility", Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT);

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

        this.pipelineManager.update("render-nodes", {
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
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
            name: "rays-generate",
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

        // Register compute passes for timestamp queries
        this.profiler.registerTimer("rays-generate");
        this.profiler.registerTimer("nodes-find");
        this.profiler.registerTimer("nodes-sort");
        this.profiler.registerTimer("collision");
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
            this.runRenderPass();
        }

        this.animationFrameId = requestAnimationFrame(this.render);
    };

    async runNodeCollision(sort: boolean = true, useProfiler: boolean = true) {
        this.bufferManager.clear("ray_nodes");
        this.bufferManager.clear("debug_distance");
        this.bufferManager.clear("node_visibility");

        const encoderFind: GPUCommandEncoder = this.device.createCommandEncoder();
        const passFind = useProfiler
            ? this.profiler.beginComputePass("nodes-find", encoderFind)
            : encoderFind.beginComputePass();

        this.computeFindLeaves(passFind);
        passFind.end();

        if (useProfiler) {
            await this.profiler.endComputePass("nodes-find", encoderFind);
        } else {
            this.device.queue.submit([encoderFind.finish()]);
        }

        if (sort) {
            const encoderSort: GPUCommandEncoder = this.device.createCommandEncoder();
            const passSort = useProfiler
                ? this.profiler.beginComputePass("nodes-sort", encoderSort)
                : encoderSort.beginComputePass(); this.computebitonicSort(passSort);

            this.computebitonicSort(passSort);
            passSort.end();

            if (useProfiler) {
                await this.profiler.endComputePass("nodes-sort", encoderSort);
            } else {
                this.device.queue.submit([encoderSort.finish()]);
            }
        }
    }

    async runGenerateRays(useProfiler: boolean = true) {
        this.bufferManager.clear("rays");

        const encoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const pass = useProfiler
            ? this.profiler.beginComputePass("rays-generate", encoder)
            : encoder.beginComputePass();

        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");
        pass.setPipeline(this.pipelineManager.get<GPUComputePipeline>("rays-generate"));
        pass.setBindGroup(0, this.bindGroupsManager.getGroup("compute-uniforms"));
        pass.setBindGroup(1, this.bindGroupsManager.getGroup("rays"));
        pass.dispatchWorkgroups(...gridLayout.dispatchSize);
        pass.end();

        if (useProfiler) {
            await this.profiler.endComputePass("rays-generate", encoder);
        } else {
            this.device.queue.submit([encoder.finish()]);
        }
    }

    async runPointCollision(useProfiler: boolean = true) {
        const encoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const pass = useProfiler
            ? this.profiler.beginComputePass("collision", encoder)
            : encoder.beginComputePass();

        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");
        pass.setPipeline(this.pipelineManager.get<GPUComputePipeline>("collision"));
        pass.setBindGroup(0, this.bindGroupsManager.getGroup("compute-uniforms"));
        pass.setBindGroup(1, this.bindGroupsManager.getGroup("rays"));
        pass.setBindGroup(2, this.bindGroupsManager.getGroup("sort"));
        pass.setBindGroup(3, this.bindGroupsManager.getGroup("collision"));
        pass.dispatchWorkgroups(...gridLayout.dispatchSize);
        pass.end();

        if (useProfiler) {
            await this.profiler.endComputePass("collision", encoder);
        } else {
            this.device.queue.submit([encoder.finish()]);
        }
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

    async runPanoramaPass() {
        const startPhi = Math.PI / 3;
        const endPhi = 2 * Math.PI / 3;
        const step = Math.PI / 180; // 1 degree

        for (let theta = 0; theta < 2 * Math.PI; theta += step) {
            if (!this.runningPanorama) {
                Utils.showToast("Panorama stopped.", "info");
                break;
            }

            let startTheta = theta;
            let endTheta = theta + step;

            this.updateThetaPhi([startTheta, endTheta], [startPhi, endPhi]);
            await this.runGenerateRays(false);
            await this.runNodeCollision(true, false);
            await this.runPointCollision(false);

            await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
        }
    }

    runRenderPass() {
        this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 64, new Float32Array(this.scene.camera.viewMatrix));
        this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 128, new Float32Array(this.scene.camera.projectionMatrix));


        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();

        if (this.canRender.points && this.renderPoints) {
            this.transparentPassDescriptor.colorAttachments[0].view = this.accumTexture.createView();
            this.transparentPassDescriptor.colorAttachments[1].view = this.revealageTexture.createView();
            this.transparentPassDescriptor.depthStencilAttachment!.view = this.depthView;

            const oitPass = commandEncoder.beginRenderPass(this.transparentPassDescriptor);
            oitPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("transparent"));
            oitPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            oitPass.setBindGroup(1, this.bindGroupsManager.getGroup("points"));
            oitPass.setVertexBuffer(0, this.bufferManager.get("points"));
            oitPass.setVertexBuffer(1, this.bufferManager.get("colors"));
            oitPass.setVertexBuffer(2, this.bufferManager.get("classification"));
            const pointsToDraw = this.bufferManager.get("points").size / 16;
            oitPass.draw(pointsToDraw, 1);
            oitPass.end();

        }

        const needsOpaquePass =
            (this.canRender.mesh && this.renderMesh) ||
            (this.canRender.rays && this.renderRays) ||
            (this.canRender.nodes && this.renderNodes);


        if (needsOpaquePass) {
            const swapchainTexture = this.context.getCurrentTexture();
            this.renderPassDescriptor.colorAttachments[0].view = swapchainTexture.createView();
            this.renderPassDescriptor.depthStencilAttachment!.view = this.depthView;

            const renderPass = commandEncoder.beginRenderPass(this.renderPassDescriptor);

            if (this.canRender.mesh && this.renderMesh) {
                // Render wireframe
                renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-wireframe"));
                renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
                renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
                renderPass.draw(4, this.scene.triangleCount); // 1 -> 2 -> 3 -> 1
            }

            // Render rays
            if (this.canRender.rays && this.renderRays) {
                renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-rays"));
                renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
                renderPass.draw(2 * this.raySamples[0] * this.raySamples[1], 1);
            }

            // Render nodes
            if (this.canRender.nodes && this.renderNodes) {
                renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-nodes"));
                renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("nodes"));
                renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
                renderPass.draw(24, QuadTree.leafNodes(this.scene.tree.depth));
            }

            renderPass.end();
        }

        const swapchainTexture = this.context.getCurrentTexture();
        const swapchainView = swapchainTexture.createView();

        if (this.canRender.points && this.renderPoints) {
            this.compositePassDescriptor.colorAttachments[0].view = swapchainView;

            const compositePass = commandEncoder.beginRenderPass(this.compositePassDescriptor);
            compositePass.setPipeline(this.pipelineManager.get("composite")); // full-screen triangle
            compositePass.setBindGroup(0, this.bindGroupsManager.getGroup("composite")); // uses accum + reveal as input
            compositePass.draw(6);
            compositePass.end();
        }

        const gizmoPassDesc: GPURenderPassDescriptor = {
            label: 'pass-gizmo',
            colorAttachments: [
                {
                    view: swapchainView,
                    loadOp: "load", // preserve composite result
                    storeOp: "store",
                }
            ]
        };

        const { gmodel, gview, gprojection } = this.scene.gizmo.getModelViewProjection(
            this.scene.camera, this.canvas.width, this.canvas.height);

        this.bufferManager.write("gizmo_uniforms", gmodel, 0);
        this.bufferManager.write("gizmo_uniforms", gview, 64);
        this.bufferManager.write("gizmo_uniforms", gprojection, 128);

        const gizmoPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(gizmoPassDesc);
        gizmoPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-gizmo"));
        gizmoPass.setBindGroup(0, this.bindGroupsManager.getGroup("gizmo"));
        gizmoPass.setVertexBuffer(0, this.bufferManager.get("gizmo_vertices"));
        gizmoPass.draw(6);
        gizmoPass.end();

        // Submit everything
        this.device.queue.submit([commandEncoder.finish()]);
    }

    updateRaySamples(samples: [number, number] = [1, 1]) {
        const [oldX, oldY] = this.raySamples;
        this.raySamples[0] = samples[0];
        this.raySamples[1] = samples[1];

        this.device.queue.writeBuffer(this.bufferManager.get("comp_uniforms"), 32, new Uint32Array([
            this.raySamples[0], this.raySamples[1],
        ]));

        if (oldX !== this.raySamples[0] || oldY !== this.raySamples[1]) {
            this.resizeRayRelatedBuffers();
            this.updateRayWorkgroups();
            this.updateRayPipelines();
        }
    }

    updateThetaPhi(theta: [number, number] = [0, 0], phi: [number, number] = [0, 0]) {
        this.device.queue.writeBuffer(
            this.bufferManager.get("comp_uniforms"), 12,
            new Float32Array([theta[0], theta[1], phi[0], phi[1]]));
    }

    updateRayOrigin(origin: [number, number, number] = [0, 0, 0]) {
        this.device.queue.writeBuffer(this.bufferManager.get("comp_uniforms"), 0, new Float32Array(origin));
    }

    updateRenderMode(mode: number = 0) {
        this.bufferManager.write("render_mode", new Uint32Array([mode]));
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

    updateRayWorkgroups() {
        this.workgroups.update("2d-grid-per-ray", {
            problemSize: [this.raySamples[0], this.raySamples[1], 1],
        });
        this.workgroups.update("workgroup-per-ray", {
            problemSize: [this.raySamples[0] * this.raySamples[1], 1, 1],
        });
    }

    updateRayPipelines() {
        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");
        const linearLayout = this.workgroups.getLayout("workgroup-per-ray");

        const gridConstants = {
            WORKGROUP_SIZE_X: gridLayout.workgroupSize[0],
            WORKGROUP_SIZE_Y: gridLayout.workgroupSize[1],
            WORKGROUP_SIZE_Z: gridLayout.workgroupSize[2],
        };

        this.pipelineManager.update("rays-generate", { codeConstants: gridConstants });
        this.pipelineManager.update("find-leaves", { codeConstants: gridConstants });
        this.pipelineManager.update("collision", { codeConstants: gridConstants });

        this.pipelineManager.update("bitonic-sort", {
            codeConstants: {
                WORKGROUP_SIZE: linearLayout.workgroupSize[0],
            }
        });
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

    clearVisibility() {
        this.bufferManager.clear("point_visibility");
    }

    reset() {
        this.canRender = {
            gizmo: true,
            rays: true,
            points: false,
            mesh: false,
            nodes: false,
        }

        this.ui?.setRunNodesButtonDisabled(true);
        this.ui?.setRunPointsButtonDisabled(true);
        this.ui?.setRunPanoramaButtonDisabled(true);

        this.stopRendering();
        this.init();
        this.startRendering();
    }

    resize(width: number, height: number) {
        this.canvas.width = width;
        this.canvas.height = height;

        this.configureContext();

        this.createDepthTexture(width, height);
        this.createOITTextures(width, height);
    }
}