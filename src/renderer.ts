import find_leaves_src from "./shaders/find-leaves.wgsl"
import bitonic_sort_src from "./shaders/bitonic-sort.wgsl"

import wireframe_src from "./shaders/wireframe.wgsl"
import points_src from "./shaders/points.wgsl"
import rays_src from "./shaders/rays.wgsl"
import gizmo_src from "./shaders/gizmo.wgsl"
import nodes_src from "./shaders/nodes.wgsl"

import { Scene } from "./scene";
import { mat3, mat4, vec3 } from "gl-matrix";


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

    // Bind groups
    bind_group_find: GPUBindGroup;
    bind_group_sort: GPUBindGroup;

    bind_group_render: GPUBindGroup;
    bind_group_gizmo: GPUBindGroup;
    bind_group_points: GPUBindGroup;
    bind_group_nodes: GPUBindGroup;

    bind_group_layout_find: GPUBindGroupLayout;
    bind_group_layout_sort: GPUBindGroupLayout;

    bind_group_layout_render: GPUBindGroupLayout;
    bind_group_layout_gizmo: GPUBindGroupLayout;
    bind_group_layout_points: GPUBindGroupLayout;
    bind_group_layout_nodes: GPUBindGroupLayout;

    raySamples: Uint32Array = new Uint32Array([16, 32]);

    // Buffers
    compUniformBuffer: GPUBuffer;
    vsUniformBuffer: GPUBuffer;
    pointBuffer: GPUBuffer;
    colorBuffer: GPUBuffer;
    indicesBuffer: GPUBuffer;
    pointVisibilityBuffer: GPUBuffer;
    nodeVisibilityBuffer: GPUBuffer;
    rayBuffer: GPUBuffer;
    gizmoVertexBuffer: GPUBuffer;
    gizmoUniformsBuffer: GPUBuffer;
    renderModeBuffer: GPUBuffer;
    nodesBuffer: GPUBuffer;
    pointToNodeBuffer: GPUBuffer;
    nodeToTriangleBuffer: GPUBuffer;
    closestHitBuffer: GPUBuffer;
    rayToNodeBuffer: GPUBuffer;
    debugDistancesBuffer: GPUBuffer;
    rayNodeCountsBuffer: GPUBuffer;

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

        const pointBufferSize = Math.min(this.scene.points.byteLength, this.device.limits.maxStorageBufferBindingSize);
        this.pointBuffer = this.device.createBuffer({
            label: 'buffer-points',
            size: pointBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.pointBuffer, 0, this.scene.points, 0, pointBufferSize / Float32Array.BYTES_PER_ELEMENT);

        const colorBufferSize = Math.min(this.scene.colors.byteLength, this.device.limits.maxStorageBufferBindingSize);
        this.colorBuffer = this.device.createBuffer({
            label: 'buffer-colors',
            size: colorBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.colorBuffer, 0, this.scene.colors, 0, colorBufferSize / Float32Array.BYTES_PER_ELEMENT);

        const indicesBufferSize = Math.min(this.scene.indices.byteLength, this.device.limits.maxStorageBufferBindingSize);
        this.indicesBuffer = this.device.createBuffer({
            label: 'buffer-index',
            size: indicesBufferSize,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.indicesBuffer, 0, this.scene.indices, 0, indicesBufferSize / Uint32Array.BYTES_PER_ELEMENT);

        const pointVisibilityBufferSize = Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT;
        this.pointVisibilityBuffer = this.device.createBuffer({
            label: 'buffer-visibility-points',
            size: pointVisibilityBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        const nodeVisibilityBufferSize = (Math.pow(4, this.scene.tree.depth) / 32) * Uint32Array.BYTES_PER_ELEMENT;
        this.nodeVisibilityBuffer = this.device.createBuffer({
            label: 'buffer-visibility-nodes',
            size: nodeVisibilityBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        this.compUniformBuffer = this.device.createBuffer({
            label: 'uniform-comp',
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.vsUniformBuffer = this.device.createBuffer({
            label: 'uniform-vs',
            size: 192,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.rayBuffer = this.device.createBuffer({
            label: 'buffer-ray',
            size: this.raySamples[0] * this.raySamples[1] * 2 * 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
        });

        this.gizmoVertexBuffer = this.device.createBuffer({
            label: 'buffer-gizmo-vertex',
            size: this.scene.gizmo.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.gizmoVertexBuffer, 0, this.scene.gizmo);

        this.gizmoUniformsBuffer = this.device.createBuffer({
            label: 'buffer-gizmo-uniforms',
            size: 192,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.renderModeBuffer = this.device.createBuffer({
            label: 'buffer-rendermode',
            size: 4, // 1 x u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const nodeBuffer = this.scene.tree.flatten();
        const pointToNodeBufferData = this.scene.tree.mapPointsToNodes();
        this.nodesBuffer = this.device.createBuffer({
            label: 'buffer-nodes',
            size: nodeBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.nodesBuffer, 0, nodeBuffer);

        this.pointToNodeBuffer = this.device.createBuffer({
            label: 'buffer-points-to-nodes',
            size: pointToNodeBufferData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.pointToNodeBuffer, 0, pointToNodeBufferData);

        this.nodeToTriangleBuffer = this.device.createBuffer({
            label: 'buffer-nodes-to-triangles',
            size: this.scene.nodeToTriangles.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.nodeToTriangleBuffer, 0, this.scene.nodeToTriangles);

        this.closestHitBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const maxFloatData = new Uint32Array([0x7F7FFFFF, 0, 0, 0]); // IEEE-754 bit pattern for f32::MAX
        this.device.queue.writeBuffer(this.closestHitBuffer, 0, maxFloatData);

        this.rayToNodeBuffer = this.device.createBuffer({
            size: this.raySamples[0] * this.raySamples[1] * 128 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false,
        });

        this.debugDistancesBuffer = this.device.createBuffer({
            size: this.raySamples[0] * this.raySamples[1] * 128 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false,
        });

        this.rayNodeCountsBuffer = this.device.createBuffer({
            size: this.raySamples[0] * this.raySamples[1] * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false,
        }); // TODO: Update the size after ray samples change!

        this.bind_group_layout_find = this.device.createBindGroupLayout({
            label: 'layout-find-leaves',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },

            ]
        });

        this.bind_group_layout_sort = this.device.createBindGroupLayout({
            label: 'layout-sort',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },

            ]
        });

        this.bind_group_layout_render = this.device.createBindGroupLayout({
            label: 'render_layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" }
                },
            ]
        });

        this.bind_group_layout_gizmo = this.device.createBindGroupLayout({
            label: 'gizmo_layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" },
                },
            ],
        });

        this.bind_group_layout_points = this.device.createBindGroupLayout({
            label: 'points_layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                }
            ]
        });

        this.bind_group_layout_nodes = this.device.createBindGroupLayout({
            label: 'layout-nodes',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" }
                },
            ]
        });

        this.bind_group_find = this.device.createBindGroup({
            label: 'bind-group-find',
            layout: this.bind_group_layout_find,
            entries: [
                { binding: 0, resource: { buffer: this.nodesBuffer } },
                { binding: 1, resource: { buffer: this.rayBuffer } },
                { binding: 2, resource: { buffer: this.rayToNodeBuffer } },
                { binding: 3, resource: { buffer: this.rayNodeCountsBuffer } },
                { binding: 4, resource: { buffer: this.nodeVisibilityBuffer } },
                { binding: 5, resource: { buffer: this.compUniformBuffer } },
            ]
        });

        this.bind_group_sort = this.device.createBindGroup({
            label: 'bind-group-sort',
            layout: this.bind_group_layout_sort,
            entries: [
                { binding: 0, resource: { buffer: this.nodesBuffer } },
                { binding: 1, resource: { buffer: this.rayBuffer } },
                { binding: 2, resource: { buffer: this.rayNodeCountsBuffer } },
                { binding: 3, resource: { buffer: this.rayToNodeBuffer } },
                { binding: 4, resource: { buffer: this.debugDistancesBuffer } },
                { binding: 5, resource: { buffer: this.compUniformBuffer } },
            ]
        });


        this.bind_group_render = this.device.createBindGroup({
            label: 'render_bind_group',
            layout: this.bind_group_layout_render,
            entries: [
                { binding: 0, resource: { buffer: this.pointBuffer } },
                { binding: 1, resource: { buffer: this.indicesBuffer } }, // Check if this is needed, as the vertex/indices buffer is not used in the point and ray shader
                { binding: 2, resource: { buffer: this.pointVisibilityBuffer } },
                { binding: 4, resource: { buffer: this.vsUniformBuffer } },
                { binding: 5, resource: { buffer: this.rayBuffer } }
            ]
        });

        this.bind_group_gizmo = this.device.createBindGroup({
            layout: this.bind_group_layout_gizmo,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.gizmoUniformsBuffer },
                },
            ],
        });

        this.bind_group_points = this.device.createBindGroup({
            layout: this.bind_group_layout_points,
            entries: [
                { binding: 0, resource: { buffer: this.renderModeBuffer } },
                { binding: 1, resource: { buffer: this.nodesBuffer } },
                { binding: 2, resource: { buffer: this.pointToNodeBuffer } }
            ]
        });

        this.bind_group_nodes = this.device.createBindGroup({
            layout: this.bind_group_layout_nodes,
            entries: [
                { binding: 0, resource: { buffer: this.vsUniformBuffer } },
                { binding: 1, resource: { buffer: this.nodesBuffer } },
                { binding: 2, resource: { buffer: this.nodeVisibilityBuffer } },
                { binding: 3, resource: { buffer: this.rayToNodeBuffer } }
            ]
        });


        const pipeline_layout_find = this.device.createPipelineLayout({
            label: 'pipeline-layout-find',
            bindGroupLayouts: [this.bind_group_layout_find]
        });
        const pipeline_layout_sort = this.device.createPipelineLayout({
            label: 'pipeline-layout-sort',
            bindGroupLayouts: [this.bind_group_layout_sort]
        });
        const pipeline_layout_render = this.device.createPipelineLayout({
            label: 'pipeline-layout-render',
            bindGroupLayouts: [this.bind_group_layout_render]
        });
        const pipeline_layout_points = this.device.createPipelineLayout({
            label: 'pipeline-layout-points',
            bindGroupLayouts: [this.bind_group_layout_render, this.bind_group_layout_points]
        });
        const pipeline_layout_gizmo = this.device.createPipelineLayout({
            label: 'pipeline-layout-gizmo',
            bindGroupLayouts: [this.bind_group_layout_gizmo]
        });
        const pipeline_layout_nodes = this.device.createPipelineLayout({
            label: 'pipeline-layout-nodes',
            bindGroupLayouts: [this.bind_group_layout_nodes]
        });

        this.pipelineFindLeaves = this.device.createComputePipeline({
            layout: pipeline_layout_find,
            compute: {
                module: this.device.createShaderModule({ code: find_leaves_src }),
                entryPoint: "main"
            }
        });

        this.pipelineBitonicSort = this.device.createComputePipeline({
            layout: pipeline_layout_sort,
            compute: {
                module: this.device.createShaderModule({
                    code: this.applyShaderConstants(bitonic_sort_src, {
                        WORKGROUP_SIZE: 128,
                    })
                }),
                entryPoint: "main",
                constants: {
                    BLOCK_SIZE: 128
                }
            }
        });

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
        const resetBuffer = (buffer: GPUBuffer) =>
            this.device.queue.writeBuffer(buffer, 0, new Uint32Array(buffer.size / 4));
        resetBuffer(this.rayToNodeBuffer);
        resetBuffer(this.nodeVisibilityBuffer);

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
        const workgroupSizeX = 8;
        const workgroupSizeY = 8;

        const dispatchX = Math.ceil(this.raySamples[0] / workgroupSizeX);
        const dispatchY = Math.ceil(this.raySamples[1] / workgroupSizeY);
        const dispatchZ = 1;

        encoder.setPipeline(this.pipelineFindLeaves);
        encoder.setBindGroup(0, this.bind_group_find);
        encoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }

    computebitonicSort(encoder: GPUComputePassEncoder) {

        const totalWorkgroups = this.raySamples[0] * this.raySamples[1]; // One workgroup per ray

        encoder.setPipeline(this.pipelineBitonicSort);
        encoder.setBindGroup(0, this.bind_group_sort);
        encoder.dispatchWorkgroups(totalWorkgroups);
    }

    runRenderPass() {
        this.device.queue.writeBuffer(this.vsUniformBuffer, 64, new Float32Array(this.scene.camera.viewMatrix));
        this.device.queue.writeBuffer(this.vsUniformBuffer, 128, new Float32Array(this.scene.camera.projectionMatrix));

        const colorTexture = this.context.getCurrentTexture();
        this.renderPassDescriptor.colorAttachments[0].view = colorTexture.createView();

        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

        // Render points
        const renderPointsCheckbox = <HTMLInputElement>document.getElementById("renderPoints");
        const renderPoints = renderPointsCheckbox.checked;
        if (renderPoints) {
            renderPass.setPipeline(this.pipelineRenderPoints);
            renderPass.setBindGroup(0, this.bind_group_render);
            renderPass.setBindGroup(1, this.bind_group_points);
            renderPass.setVertexBuffer(0, this.pointBuffer);
            renderPass.setVertexBuffer(1, this.colorBuffer);
            const pointsToDraw = this.pointBuffer.size / 16;
            renderPass.draw(pointsToDraw, 1);
        } else {
            renderPass.setPipeline(this.pipelineRenderWireframe);
            renderPass.setBindGroup(0, this.bind_group_render);
            renderPass.draw(4, this.scene.triangleCount); // 1 -> 2 -> 3 -> 1
        }

        // Render rays
        renderPass.setPipeline(this.pipelineRenderRays);
        renderPass.setBindGroup(0, this.bind_group_render);
        renderPass.draw(2 * this.raySamples[0] * this.raySamples[1], 1);

        // Render nodes
        const showNodesCheckbox = <HTMLInputElement>document.getElementById("showNodes");
        const showNodes = showNodesCheckbox.checked;
        if (showNodes) {
            renderPass.setPipeline(this.pipelineRenderNodes);
            renderPass.setBindGroup(0, this.bind_group_nodes);
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

        this.device.queue.writeBuffer(this.gizmoUniformsBuffer, 0, new Float32Array(gizmoModel));
        this.device.queue.writeBuffer(this.gizmoUniformsBuffer, 64, new Float32Array(gizmoView));
        this.device.queue.writeBuffer(this.gizmoUniformsBuffer, 128, new Float32Array(gizmoProjection));

        const gizmoPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(gizmoPassDescriptor);
        gizmoPass.setPipeline(this.pipelineRenderGizmo);
        gizmoPass.setBindGroup(0, this.bind_group_gizmo);
        gizmoPass.setVertexBuffer(0, this.gizmoVertexBuffer);
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

        this.device.queue.writeBuffer(this.compUniformBuffer, 0, new Float32Array([
            originX, originY, originZ, // 12
            startTheta, endTheta, startPhi, endPhi // 32
        ]));
        this.device.queue.writeBuffer(this.compUniformBuffer, 32, new Uint32Array([
            this.raySamples[0], this.raySamples[1],  // 40
        ]));

        this.device.queue.writeBuffer(this.renderModeBuffer, 0, new Uint32Array([renderMode]));

        if (oldSamples[0] != this.raySamples[0] || oldSamples[1] != this.raySamples[1]) {
            this.rayBuffer = this.device.createBuffer({
                label: 'buffer-ray',
                size: this.raySamples[0] * this.raySamples[1] * 2 * 4 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
            });
            // Update the bind group with the new ray buffer
            this.bind_group_find = this.device.createBindGroup({
                label: 'bind-group-find',
                layout: this.bind_group_layout_find,
                entries: [
                    { binding: 0, resource: { buffer: this.nodesBuffer } },
                    { binding: 1, resource: { buffer: this.rayBuffer } },
                    { binding: 2, resource: { buffer: this.rayToNodeBuffer } },
                    { binding: 3, resource: { buffer: this.rayNodeCountsBuffer } },
                    { binding: 4, resource: { buffer: this.nodeVisibilityBuffer } },
                    { binding: 5, resource: { buffer: this.compUniformBuffer } },
                ]
            });

            this.bind_group_sort = this.device.createBindGroup({
                label: 'bind-group-sort',
                layout: this.bind_group_layout_sort,
                entries: [
                    { binding: 0, resource: { buffer: this.nodesBuffer } },
                    { binding: 1, resource: { buffer: this.rayBuffer } },
                    { binding: 2, resource: { buffer: this.rayNodeCountsBuffer } },
                    { binding: 3, resource: { buffer: this.rayToNodeBuffer } },
                    { binding: 4, resource: { buffer: this.debugDistancesBuffer } },
                    { binding: 5, resource: { buffer: this.compUniformBuffer } },
                ]
            });

            this.bind_group_render = this.device.createBindGroup({
                label: 'render_bind_group',
                layout: this.bind_group_layout_render,
                entries: [
                    { binding: 0, resource: { buffer: this.pointBuffer } },
                    { binding: 1, resource: { buffer: this.indicesBuffer } },
                    { binding: 2, resource: { buffer: this.pointVisibilityBuffer } },
                    { binding: 4, resource: { buffer: this.vsUniformBuffer } },
                    { binding: 5, resource: { buffer: this.rayBuffer } }
                ]
            });

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