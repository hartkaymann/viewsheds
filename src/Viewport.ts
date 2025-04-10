
import wireframe_src from "./shaders/render/wireframe.wgsl"
import points_src from "./shaders/render/points.wgsl"
import rays_src from "./shaders/render/rays.wgsl"
import gizmo_src from "./shaders/render/gizmo.wgsl"
import nodes_src from "./shaders/render/nodes.wgsl"
import composite_src from "./shaders/render/composite.wgsl"

import { vec3 } from "gl-matrix";
import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Camera } from "./Camera";
import { Gizmo } from "./Gizmo";
import { InputHandler } from "./InputHandler";
import { Scene } from "./Scene";
import { PipelineManager } from "./PipelineManager";
import { QuadTree } from "./Optimization"
import { RenderPlan } from "./Controller"

export class Viewport {
  device: GPUDevice;
  scene: Scene;
  bufferManager: BufferManager;
  bindGroupManager: BindGroupManager;
  pipelineManager: PipelineManager;

  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;

  camera: Camera;
  input: InputHandler;
  gizmo: Gizmo | null = null;

  // Device/Context objects
  renderPassDescriptor: GPURenderPassDescriptor;
  transparentPassDescriptor: GPURenderPassDescriptor;
  compositePassDescriptor: GPURenderPassDescriptor;

  //Assets
  depthTexture: GPUTexture;
  accumTexture: GPUTexture;
  revealageTexture: GPUTexture;
  depthView: GPUTextureView;

  constructor(device: GPUDevice, scene: Scene, buffers: BufferManager, bind: BindGroupManager) {
    this.device = device;
    this.scene = scene;
    this.bufferManager = buffers;
    this.bindGroupManager = bind;

    this.pipelineManager = new PipelineManager(this.device);

    this.canvas = <HTMLCanvasElement>document.getElementById("gfx-main");

    this.camera = new Camera(
      [10, 10, 10],
      [0, 0, 0],
      [0, 1, 0],
      Math.PI / 4,
      this.canvas.width / this.canvas.height,
      0.1,
      10000
    );

    const wrapper = document.getElementById('canvas-wrapper')!;
    const devicePixelRatio = window.devicePixelRatio || 1;
    const updateCanvasSize = () => {
      const width = Math.floor(wrapper.clientWidth * devicePixelRatio);
      const height = Math.floor(wrapper.clientHeight * devicePixelRatio);

      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.resize(width, height);
      }

      this.camera.aspect = width / height;
      this.camera.setProjection();
    };

    const resizeObserver = new ResizeObserver(updateCanvasSize);
    resizeObserver.observe(wrapper);

    this.input = new InputHandler(this.canvas, this.camera);

    this.gizmo = new Gizmo();
  }

  async init() {
    this.context = <GPUCanvasContext>this.canvas.getContext("webgpu");
    this.format = "bgra8unorm";

    this.configureContext();

    this.bufferManager.initBuffers([
      {
        name: "vs_uniforms",
        size: 192,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      {
        name: "gizmo_vertices",
        size: this.gizmo.vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        data: this.gizmo.vertices
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
    ]);

    this.bindGroupManager.createLayout({
      name: "points",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "render",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ]
    });



    this.bindGroupManager.createLayout({
      name: "nodes",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "composite",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "gizmo",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ]
    });

    this.createDepthTexture(this.context.canvas.width, this.context.canvas.height);
    this.createOITTextures(this.context.canvas.width, this.context.canvas.height);



    this.bindGroupManager.createGroup({
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


    this.bindGroupManager.createGroup({
      name: "points",
      layoutName: "points",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("render_mode") } },
        { binding: 1, resource: { buffer: this.bufferManager.get("nodes") } },
        { binding: 2, resource: { buffer: this.bufferManager.get("point_to_node") } }
      ]
    });


    this.bindGroupManager.createGroup({
      name: "nodes",
      layoutName: "nodes",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("vs_uniforms") } },
        { binding: 1, resource: { buffer: this.bufferManager.get("nodes") } },
        { binding: 2, resource: { buffer: this.bufferManager.get("node_visibility") } },
        { binding: 3, resource: { buffer: this.bufferManager.get("ray_nodes") } }
      ]
    });

    this.bindGroupManager.createGroup({
      name: "gizmo",
      layoutName: "gizmo",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("gizmo_uniforms") } },
      ]
    });

    this.bindGroupManager.createGroup({
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
      bindGroupLayouts: this.bindGroupManager.getLayouts(["render"])
    });
    const pipeline_layout_points = this.device.createPipelineLayout({
      label: 'pipeline-layout-points',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["render", "points"])
    });
    const pipeline_layout_gizmo = this.device.createPipelineLayout({
      label: 'pipeline-layout-gizmo',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["gizmo"])
    });
    const pipeline_layout_nodes = this.device.createPipelineLayout({
      label: 'pipeline-layout-nodes',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["nodes"])
    });
    const pipeline_layout_composite = this.device.createPipelineLayout({
      label: 'pipeline-layout-composite',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["composite"])
    });

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
                srcFactor: "one",
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
        depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
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
          loadOp: 'load',
          storeOp: 'store',
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
    this.updateRenderMode();

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

    if (!this.bindGroupManager.getGroup("composite"))
      return;

    this.bindGroupManager.updateGroup("composite", [
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

  private configureContext() {
    this.context?.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque",
    });
  }

  runRenderPass(plan: RenderPlan) {
    this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 64, new Float32Array(this.camera.viewMatrix));
    this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 128, new Float32Array(this.camera.projectionMatrix));


    const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
    const swapchainTexture = this.context.getCurrentTexture();
    const swapchainView = swapchainTexture.createView();
    this.compositePassDescriptor.colorAttachments[0].view = swapchainView;
    this.renderPassDescriptor.colorAttachments[0].view = swapchainTexture.createView();
    this.renderPassDescriptor.depthStencilAttachment!.view = this.depthView;

    if (plan.transparent) {
      // Transparent pass (accum + revealage targets)
      this.transparentPassDescriptor.colorAttachments[0].view = this.accumTexture.createView();
      this.transparentPassDescriptor.colorAttachments[1].view = this.revealageTexture.createView();
      this.transparentPassDescriptor.depthStencilAttachment!.view = this.depthView;

      const oitPass = commandEncoder.beginRenderPass(this.transparentPassDescriptor);
      oitPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("transparent"));
      oitPass.setBindGroup(0, this.bindGroupManager.getGroup("render"));
      oitPass.setBindGroup(1, this.bindGroupManager.getGroup("points"));
      oitPass.setVertexBuffer(0, this.bufferManager.get("points"));
      oitPass.setVertexBuffer(1, this.bufferManager.get("colors"));
      oitPass.setVertexBuffer(2, this.bufferManager.get("classification"));
      const pointsToDraw = this.bufferManager.get("points").size / 16;
      oitPass.draw(pointsToDraw, 1);
      oitPass.end();

      // Composite pass (writes to swapchain)
      const compositePass = commandEncoder.beginRenderPass(this.compositePassDescriptor);
      compositePass.setPipeline(this.pipelineManager.get("composite")); // full-screen triangle
      compositePass.setBindGroup(0, this.bindGroupManager.getGroup("composite")); // uses accum + reveal as input
      compositePass.draw(6);
      compositePass.end();
    }

    const needsOpaquePass =
      plan.mesh ||
      plan.rays ||
      plan.nodes;


    if (needsOpaquePass) {
      const renderPass = commandEncoder.beginRenderPass(this.renderPassDescriptor);

      if (plan.mesh) {
        renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-wireframe"));
        renderPass.setBindGroup(0, this.bindGroupManager.getGroup("render"));
        renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
        renderPass.draw(4, this.scene.triangleCount); // 1 -> 2 -> 3 -> 1
      }

      // Render rays
      if (plan.rays) {
        renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-rays"));
        renderPass.setBindGroup(0, this.bindGroupManager.getGroup("render"));
        renderPass.draw(2 * this.scene.rays.samples[0] * this.scene.rays.samples[1], 1);
      }

      // Render nodes
      if (plan.nodes) {
        renderPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-nodes"));
        renderPass.setBindGroup(0, this.bindGroupManager.getGroup("nodes"));
        renderPass.setVertexBuffer(0, this.bufferManager.get("points"));
        renderPass.draw(24, QuadTree.leafNodes(this.scene.tree.depth));
      }

      renderPass.end();
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

    const { gmodel, gview, gprojection } = this.gizmo.getModelViewProjection(
      this.camera, this.canvas.width, this.canvas.height);

    this.bufferManager.write("gizmo_uniforms", gmodel, 0);
    this.bufferManager.write("gizmo_uniforms", gview, 64);
    this.bufferManager.write("gizmo_uniforms", gprojection, 128);

    const gizmoPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(gizmoPassDesc);
    gizmoPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-gizmo"));
    gizmoPass.setBindGroup(0, this.bindGroupManager.getGroup("gizmo"));
    gizmoPass.setVertexBuffer(0, this.bufferManager.get("gizmo_vertices"));
    gizmoPass.draw(6);
    gizmoPass.end();

    // Submit everything
    this.device.queue.submit([commandEncoder.finish()]);
  }


  focusCameraOnPointCloud() {
    const bounds = this.scene.bounds;

    const centerX = (bounds.min.x + bounds.max.x) / 2;
    const centerY = (bounds.min.y + bounds.max.y) / 2;
    const centerZ = (bounds.min.z + bounds.max.z) / 2;

    // Move the camera back along the Z-axis to fit the whole cloud in view
    const distance = Math.max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z) * 1.5;

    this.camera.setPosition(vec3.fromValues(centerX, centerY, centerZ + distance));
    this.camera.setTarget(vec3.fromValues(centerX, centerY, centerZ));
  }
  clearVisibility() {
    this.bufferManager.clear("point_visibility");
  }

  updateRenderMode(mode: number = 0) {
    this.bufferManager.write("render_mode", new Uint32Array([mode]));
  }

  resize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;

    this.configureContext();

    this.createDepthTexture(width, height);
    this.createOITTextures(width, height);
  }
}