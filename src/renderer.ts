import find_leaves_src from "./shaders/find-leaves.wgsl"
import bitonic_sort_src from "./shaders/bitonic-sort.wgsl"

import wireframe_src from "./shaders/wireframe.wgsl"
import points_src from "./shaders/points.wgsl"
import rays_src from "./shaders/rays.wgsl"
import gizmo_src from "./shaders/gizmo.wgsl"
import nodes_src from "./shaders/nodes.wgsl"

import { Scene } from "./Scene";
import { mat3, mat4, vec3 } from "gl-matrix";
import { WorkgroupLayout, WorkgroupLimits, WorkgroupStrategy } from "./types/types"
import { BufferManager } from "./BufferManager"
import { QuadTree } from "./Optimization"
import { BindGroupManager } from "./BindGroupsManager"
import { PipelineManager } from "./PipelineManager"
import { WorkgroupManager } from "./WorkgroupManager"


export class Renderer {

    canvas: HTMLCanvasElement

    // Device/Context objects
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    renderPassDescriptor: GPURenderPassDescriptor;

    //Assets
    depth_buffer: GPUTexture;
    depth_buffer_view: GPUTextureView;

    raySamples: Uint32Array = new Uint32Array([1, 1]);

    // Managers
    bufferManager: BufferManager;
    bindGroupsManager: BindGroupManager;
    pipelineManager: PipelineManager;
    workgroups: WorkgroupManager;

    // Scene to render
    scene: Scene

    canRender = {
        gizmo: true,
        points: false,
        nodes: false,
        mesh: false,
    };

    // Time
    prevTime = 0;
    timeAccumulator = 0;
    readonly timeStep = 1 / 60;
    fps = 0;
    fpsLastTime = 0;
    frameCount = 0;

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

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: "opaque"
        });

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
                name: "indices",
                size: this.scene.indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "point_visibility",
                size: Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
        ]);

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

    setPointData() {
        this.bufferManager.resize("points", this.scene.points.byteLength);
        this.bufferManager.write("points", this.scene.points);

        this.bufferManager.resize("colors", this.scene.colors.byteLength);
        this.bufferManager.write("colors", this.scene.colors);

        this.bufferManager.resize("point_visibility", Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT);

        this.bindGroupsManager.updateGroup("render", [
            { binding: 0, resource: { buffer: this.bufferManager.get("points") } },
            { binding: 2, resource: { buffer: this.bufferManager.get("point_visibility") } },
        ]);

        this.canRender.points = true;
    }

    setNodeData() {
        this.bufferManager.resize("nodes", QuadTree.totalNodes(this.scene.tree.depth) * QuadTree.BYTES_PER_NODE * 4);
        this.bufferManager.write("nodes", this.scene.tree.flatten());

        this.bufferManager.resize("point_to_node", this.scene.tree.root.pointCount * 4);
        this.bufferManager.write("point_to_node", this.scene.tree.mapPointsToNodes());

        this.bufferManager.resize("node_visibility", QuadTree.leafNodes(this.scene.tree.depth) / 32 * Uint32Array.BYTES_PER_ELEMENT);
        this.bufferManager.resize("ray_nodes", this.raySamples[0] * this.raySamples[1] * QuadTree.noMaxNodesHit(this.scene.tree.depth) * 4);

        this.bindGroupsManager.updateGroup("find", [
            { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
            { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
            { binding: 4, resource: { buffer: this.bufferManager.get("node_visibility") } },
        ]);

        this.bindGroupsManager.updateGroup("sort", [
            { binding: 0, resource: { buffer: this.bufferManager.get("nodes") } },
            { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } },
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
            codeConstants: {
                MAX_STACK_SIZE: 2 * this.scene.tree.depth + 1
            },
            constants: {
                TREE_DEPTH: this.scene.tree.depth,
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            }
        });

        this.workgroups.update("workgroup-per-ray", {
            strategyArgs: [QuadTree.noMaxNodesHit(this.scene.tree.depth)],
        });
        const linearWorkgroup = this.workgroups.getLayout("workgroup-per-ray");

        this.pipelineManager.update("bitonic-sort", {
            codeConstants: {
                WORKGROUP_SIZE: linearWorkgroup.workgroupSize[0],
            },
            constants: {
                BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
            }
        });

        this.canRender.nodes = true;
    }

    setMeshData() {
        this.bufferManager.resize("indices", this.scene.indices.byteLength);
        this.bufferManager.write("indices", this.scene.indices);

        this.bufferManager.resize("node_to_triangle", this.scene.nodeToTriangles.byteLength);
        this.bufferManager.write("node_to_triangle", this.scene.nodeToTriangles);

        this.bindGroupsManager.updateGroup("render", [
            { binding: 1, resource: { buffer: this.bufferManager.get("indices") } },
        ]);


        this.canRender.mesh = true;
    }

    private setupComputePipelines(): void {
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
        const tile2DGridPerRay = (): WorkgroupStrategy =>
            ({ totalThreads, problemSize }) => {
                const maxX = Math.floor(Math.sqrt(totalThreads));
                const x = Math.min(problemSize[0], maxX);
                const y = Math.min(problemSize[1], Math.floor(totalThreads / x));

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

        this.workgroups.register({
            name: "workgroup-per-ray",
            problemSize: [this.raySamples[0] * this.raySamples[1], 1, 1],
            strategyFn: oneWorkgroupPerRay,
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
        const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");
        const dispatchX = gridLayout.dispatchSize[0];
        const dispatchY = gridLayout.dispatchSize[1];
        const dispatchZ = gridLayout.dispatchSize[2];

        encoder.setPipeline(this.pipelineManager.get<GPUComputePipeline>("find-leaves"));
        encoder.setBindGroup(0, this.bindGroupsManager.getGroup("find"));
        encoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }

    computebitonicSort(encoder: GPUComputePassEncoder) {
        const linearLayout = this.workgroups.getLayout("workgroup-per-ray");
        const totalWorkgroups = linearLayout.dispatchSize[0];

        encoder.setPipeline(this.pipelineManager.get<GPUComputePipeline>("bitonic-sort"));
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

        const renderPointsCheckbox = <HTMLInputElement>document.getElementById("renderPoints");
        const renderPoints = renderPointsCheckbox.checked;
        if (this.canRender.points && renderPoints) {
            // Render points
            renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-points"));
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            renderPass.setBindGroup(1, this.bindGroupsManager.getGroup("points"));
            renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
            renderPass.setVertexBuffer(1, this.bufferManager.get("colors"));
            const pointsToDraw = this.bufferManager.get("points").size / 16;
            renderPass.draw(pointsToDraw, 1);
        } else if (this.canRender.mesh) {
            // Render wireframe
            renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-wireframe"));
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
            renderPass.draw(4, this.scene.triangleCount); // 1 -> 2 -> 3 -> 1
        }

        // Render rays
        renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-rays"));
        renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("render"));
        renderPass.draw(2 * this.raySamples[0] * this.raySamples[1], 1);

        // Render nodes
        const showNodesCheckbox = <HTMLInputElement>document.getElementById("showNodes");
        const showNodes = showNodesCheckbox.checked;
        if (this.canRender.nodes && showNodes) {
            renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-nodes"));
            renderPass.setBindGroup(0, this.bindGroupsManager.getGroup("nodes"));
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

        const { gmodel, gview, gprojection } = this.scene.gizmo.getModelViewProjection(this.scene.camera);

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
            this.bufferManager.resize("ray_nodes", this.raySamples[0] * this.raySamples[1] * (2 ** (this.scene.tree.depth + 1)) * 4);
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

            this.workgroups.update("2d-grid-per-ray", {
                problemSize: [this.raySamples[0], this.raySamples[1], 1],
            });
            this.workgroups.update("workgroup-per-ray", {
                problemSize: [this.raySamples[0] * this.raySamples[1], 1, 1],
            });
            const linearLayout = this.workgroups.getLayout("workgroup-per-ray");
            const gridLayout = this.workgroups.getLayout("2d-grid-per-ray");

            this.pipelineManager.update("find-leaves", {
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

            this.setupComputePipelines();
        }
        this.runComputePass();
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