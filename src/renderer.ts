import compute_src from "./shaders/compute.wgsl"
import triangles_src from "./shaders/triangles.wgsl"
import points_src from "./shaders/points.wgsl"
import rays_src from "./shaders/rays.wgsl"
import gizmo_src from "./shaders/gizmo.wgsl"

import { Scene } from "./scene";
import { mat3, mat4, quat, vec2, vec3 } from "gl-matrix";


export class Renderer {

    canvas: HTMLCanvasElement

    // Device/Context objects
    adapter: GPUAdapter;
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    renderPassDescriptor: GPURenderPassDescriptor;

    //Assets
    color_buffer: GPUTexture;
    color_buffer_view: GPUTextureView;
    sampler: GPUSampler;
    sceneParameters: GPUBuffer;

    // Pipeline objects
    pipelineCompute: GPUComputePipeline;
    pipelineRenderPoints: GPURenderPipeline;
    pipelineRenderTriangles: GPURenderPipeline;
    pipelineRenderRays: GPURenderPipeline;
    pipelineRenderGizmo: GPURenderPipeline;

    // Bind groups
    bind_group_compute: GPUBindGroup;
    bind_group_render: GPUBindGroup;
    bind_group_gizmo: GPUBindGroup;
    bind_group_layout_compute: GPUBindGroupLayout;
    bind_group_layout_render: GPUBindGroupLayout;
    bind_group_layout_gizmo: GPUBindGroupLayout;

    raySamples: Uint32Array = new Uint32Array([16, 32]);

    // Buffers
    compUniformBuffer: GPUBuffer;
    vsUniformBuffer: GPUBuffer;
    pointBuffer: GPUBuffer;
    colorBuffer: GPUBuffer;
    indicesBuffer: GPUBuffer;
    visibilityBuffer: GPUBuffer;
    rayBuffer: GPUBuffer;
    gizmoVertexBuffer: GPUBuffer;

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

        // adapter: wrapper around (physical) GPU.
        // Describes features and limits
        this.adapter = <GPUAdapter>await navigator.gpu?.requestAdapter();
        // device: wrapper around GPU functionality
        // Function calls are made through the device
        this.device = <GPUDevice>await this.adapter?.requestDevice();

        this.context = <GPUCanvasContext>this.canvas.getContext("webgpu");
        this.format = "bgra8unorm";

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: "opaque"
        });

        this.pointBuffer = this.device.createBuffer({
            label: 'buffer-points',
            size: this.scene.points.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.pointBuffer, 0, this.scene.points);

        this.colorBuffer = this.device.createBuffer({
            label: 'buffer-colors',
            size: this.scene.colors.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.colorBuffer, 0, this.scene.colors);

        this.indicesBuffer = this.device.createBuffer({
            label: 'buffer-index',
            size: this.scene.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.indicesBuffer, 0, this.scene.indices);

        const visibilityBufferSize = Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT;
        this.visibilityBuffer = this.device.createBuffer({
            label: 'buffer-visibility',
            size: visibilityBufferSize,
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
            size: this.scene.gizmo.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.gizmoVertexBuffer, 0, this.scene.gizmo);

        this.bind_group_layout_compute = this.device.createBindGroupLayout({
            label: 'comp_layout',
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
                    buffer: { type: "storage" }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
            ]
        });

        this.bind_group_layout_render = this.device.createBindGroupLayout({
            label: 'render_layout',
            entries: [
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
            entries: [
                {
                    binding: 0, // View matrix
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" },
                },
            ],
        });

        this.bind_group_compute = this.device.createBindGroup({
            label: 'comp_bind_group',
            layout: this.bind_group_layout_compute,
            entries: [
                { binding: 0, resource: { buffer: this.pointBuffer } },
                { binding: 1, resource: { buffer: this.indicesBuffer } },
                { binding: 2, resource: { buffer: this.visibilityBuffer } },
                { binding: 3, resource: { buffer: this.compUniformBuffer } },
                { binding: 5, resource: { buffer: this.rayBuffer } }
            ]
        });

        this.bind_group_render = this.device.createBindGroup({
            label: 'render_bind_group',
            layout: this.bind_group_layout_render,
            entries: [
                { binding: 2, resource: { buffer: this.visibilityBuffer } },
                { binding: 4, resource: { buffer: this.vsUniformBuffer } },
                { binding: 5, resource: { buffer: this.rayBuffer } }
            ]
        });

        this.bind_group_gizmo = this.device.createBindGroup({
            layout: this.bind_group_layout_gizmo,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.vsUniformBuffer },
                },
            ],
        });

        const pipeline_layout_compute = this.device.createPipelineLayout({
            label: 'compute-layout',
            bindGroupLayouts: [this.bind_group_layout_compute]
        });
        const pipeline_layout_render = this.device.createPipelineLayout({
            label: 'render-layout',
            bindGroupLayouts: [this.bind_group_layout_render]
        });
        const pipeline_layout_gizmo = this.device.createPipelineLayout({
            label: 'gizmo-layout',
            bindGroupLayouts: [this.bind_group_layout_gizmo]
        });

        const computeShaderModule = this.device.createShaderModule({ code: compute_src });
        this.pipelineCompute = this.device.createComputePipeline({
            label: 'compute-pipeline',
            layout: pipeline_layout_compute,
            compute: {
                module: computeShaderModule,
                entryPoint: 'main'
            }
        });

        const pointShaderModule = this.device.createShaderModule({ code: points_src });
        this.pipelineRenderPoints = this.device.createRenderPipeline({
            label: 'render-pipeline',
            layout: pipeline_layout_render,
            vertex: {
                module: pointShaderModule,
                entryPoint: 'main',
                buffers: [
                    {
                        arrayStride: 4 * 4, // 4 floats per position
                        attributes: [
                            {
                                shaderLocation: 0, // Position
                                offset: 0,
                                format: 'float32x4'
                            }
                        ]
                    },
                    {
                        arrayStride: 4 * 4, // 4 floats per color
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
            }
        })

        const triangleShaderModule = this.device.createShaderModule({ code: triangles_src });
        this.pipelineRenderTriangles = this.device.createRenderPipeline({
            label: 'render-pipeline',
            layout: pipeline_layout_render,
            vertex: {
                module: triangleShaderModule,
                entryPoint: 'main',
                buffers: [
                    {
                        arrayStride: 4 * 4, // 4 floats per vertex, 4 bytes per float
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x4'
                            }
                        ]
                    }
                ]
            },
            fragment: {
                module: triangleShaderModule,
                entryPoint: 'main_fs',
                targets: [
                    { format: 'bgra8unorm' } // presentationFormat
                ]
            },
            primitive: {
                topology: 'triangle-list',
                // cullMode: 'back',
                // frontFace: 'ccw'
            }
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
            }
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
                        arrayStride: 4 * 4, // 3 floats per vertex (position)
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
                topology: 'line-list',
                cullMode: 'none',
            }
        });

        this.renderPassDescriptor = {
            colorAttachments: [
                {
                    view: undefined,
                    resolveTarget: undefined,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0.12, g: 0.12, b: 0.13, a: 1.0 }
                }
            ]
        }

        const updateButton = document.getElementById("updateValues");
        updateButton.addEventListener("click", this.updateValues.bind(this));
        this.updateValues();

        this.runComputePass();
        this.render();
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

    runComputePass() {
        const computeEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const computePass: GPUComputePassEncoder = computeEncoder.beginComputePass();
        computePass.setPipeline(this.pipelineCompute);
        computePass.setBindGroup(0, this.bind_group_compute);
        computePass.dispatchWorkgroups(this.raySamples[0], this.raySamples[1], 1);
        computePass.end();
        this.device.queue.submit([computeEncoder.finish()]);
    }

    runRenderPass() {
        // this.device.queue.writeBuffer(this.vsUniformBuffer, 64, new Float32Array(this.scene.camera.viewMatrix));
        // this.device.queue.writeBuffer(this.vsUniformBuffer, 128, new Float32Array(this.scene.camera.projectionMatrix));

        const colorTexture = this.context.getCurrentTexture();
        // this.renderPassDescriptor.colorAttachments[0].view = colorTexture.createView();

        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        // const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        // const renderPointsCheckbox = <HTMLInputElement>document.getElementById("renderPoints");
        // const renderPoints = renderPointsCheckbox.checked;

        // if (renderPoints) {
        //     renderPass.setPipeline(this.pipelineRenderPoints);
        //     renderPass.setBindGroup(0, this.bind_group_render);
        //     renderPass.setVertexBuffer(0, this.pointBuffer); // Set the vertex buffer
        //     renderPass.setVertexBuffer(1, this.colorBuffer); // Set the color buffer
        //     renderPass.draw(this.scene.points.length / 4, 1);
        // } else {
        //     renderPass.setPipeline(this.pipelineRenderTriangles);
        //     renderPass.setBindGroup(0, this.bind_group_render);
        //     renderPass.setVertexBuffer(0, this.pointBuffer);
        //     renderPass.setIndexBuffer(this.indicesBuffer, "uint32");
        //     renderPass.drawIndexed(this.scene.indices.length, 1, 0, 0, 0);
        // }

        // // Render rays
        // renderPass.setPipeline(this.pipelineRenderRays);
        // renderPass.setBindGroup(0, this.bind_group_render);
        // renderPass.draw(2 * this.raySamples[0] * this.raySamples[1], 1);
        // renderPass.end();

        // Render gizmo
        let rotationMatrix = mat4.create();
        mat4.fromQuat(rotationMatrix, mat4.getRotation(quat.create(), this.scene.camera.viewMatrix));

        let gizmoModel = mat4.create();
        mat4.translate(gizmoModel, gizmoModel, vec3.fromValues(0.8, 0.8, 0.0));
        mat4.multiply(gizmoModel, gizmoModel, rotationMatrix);
        mat4.scale(gizmoModel, gizmoModel, vec3.fromValues(0.2, 0.2, 0.2));

        let gizmoView = mat4.create();
        
        let gizmoProjection = mat4.create();
        mat4.ortho(gizmoProjection, -this.scene.camera.aspect, this.scene.camera.aspect, -1, 1, -1, 1);

        this.device.queue.writeBuffer(this.vsUniformBuffer, 0, new Float32Array(gizmoModel));
        this.device.queue.writeBuffer(this.vsUniformBuffer, 64, new Float32Array(gizmoView));
        this.device.queue.writeBuffer(this.vsUniformBuffer, 128, new Float32Array(gizmoProjection));

        this.renderPassDescriptor.colorAttachments[0].view = colorTexture.createView();

        const gizmoPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
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
        const maxSteps = parseFloat((<HTMLInputElement>document.getElementById("maxSteps")).value);
        const stepSize = parseFloat((<HTMLInputElement>document.getElementById("stepSize")).value);
        const startTheta = parseFloat((<HTMLInputElement>document.getElementById("startTheta")).value);
        const endTheta = parseFloat((<HTMLInputElement>document.getElementById("endTheta")).value);
        const startPhi = parseFloat((<HTMLInputElement>document.getElementById("startPhi")).value);
        const endPhi = parseFloat((<HTMLInputElement>document.getElementById("endPhi")).value);

        this.device.queue.writeBuffer(this.compUniformBuffer, 0, new Float32Array([
            originX, originY, originZ, stepSize, // 16
            startTheta, endTheta, startPhi, endPhi // 32
        ]));
        this.device.queue.writeBuffer(this.compUniformBuffer, 32, new Uint32Array([
            this.raySamples[0], this.raySamples[1], maxSteps // 44
        ]));

        if (oldSamples[0] != this.raySamples[0] || oldSamples[1] != this.raySamples[1]) {
            this.rayBuffer = this.device.createBuffer({
                label: 'buffer-ray',
                size: this.raySamples[0] * this.raySamples[1] * 2 * 4 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
            });
            // Update the bind group with the new ray buffer
            this.bind_group_compute = this.device.createBindGroup({
                label: 'comp_bind_group',
                layout: this.bind_group_layout_compute,
                entries: [
                    { binding: 0, resource: { buffer: this.pointBuffer } },
                    { binding: 1, resource: { buffer: this.indicesBuffer } },
                    { binding: 2, resource: { buffer: this.visibilityBuffer } },
                    { binding: 3, resource: { buffer: this.compUniformBuffer } },
                    { binding: 5, resource: { buffer: this.rayBuffer } }
                ]
            });

            this.bind_group_render = this.device.createBindGroup({
                label: 'render_bind_group',
                layout: this.bind_group_layout_render,
                entries: [
                    { binding: 2, resource: { buffer: this.visibilityBuffer } },
                    { binding: 4, resource: { buffer: this.vsUniformBuffer } },
                    { binding: 5, resource: { buffer: this.rayBuffer } }
                ]
            });

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