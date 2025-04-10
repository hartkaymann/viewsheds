import generate_rays_src from "./shaders/compute/ray-gen.wgsl"
import find_leaves_src from "./shaders/compute/find-leaves.wgsl"
import bitonic_sort_src from "./shaders/compute/bitonic-sort.wgsl"
import collision_src from "./shaders/compute/collision.wgsl"

import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { QuadTree } from "./Optimization";
import { PipelineManager } from "./PipelineManager";
import { Profiler } from "./Profiler";
import { Scene } from "./Scene";
import { WorkgroupStrategy } from "./types/types";
import { WorkgroupManager } from "./WorkgroupManager";
import { Utils } from "./Utils";

export class CollisionSystem {
  device: GPUDevice;
  scene: Scene;

  bufferManager: BufferManager;
  bindGroupManager: BindGroupManager;
  pipelineManager: PipelineManager;

  profiler: Profiler;

  runningPanorama: boolean = false;
  workgroups: WorkgroupManager;


  constructor(device: GPUDevice, scene: Scene, buffers: BufferManager, binds: BindGroupManager) {
    this.device = device;
    this.scene = scene;
    this.bufferManager = buffers;
    this.bindGroupManager = binds;

    this.workgroups = new WorkgroupManager(this.device);
    this.pipelineManager = new PipelineManager(this.device);

    this.profiler = new Profiler(this.device);
    this.profiler.setBufferManager(this.bufferManager);
  }


  async init() {
    this.setupComputePipelines();
    await this.runNodeCollision();
  }

  private setupComputePipelines(): void {
    // === Pipeline Layouts ===

    const pipelineLayoutRays = this.device.createPipelineLayout({
      label: 'pipeline-layout-rays',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["ray-uniforms", "rays"]),
    });

    const pipelineLayoutFind = this.device.createPipelineLayout({
      label: 'pipeline-layout-find',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["ray-uniforms", "rays", "find"]),
    });

    const pipelineLayoutSort = this.device.createPipelineLayout({
      label: 'pipeline-layout-sort',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["ray-uniforms", "rays", "sort"]),
    });

    const pipelineLayoutCollision = this.device.createPipelineLayout({
      label: 'pipeline-layout-collision',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["ray-uniforms", "rays", "sort", "collision"]),
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
      problemSize: [this.scene.rays.samples[0], this.scene.rays.samples[1], 1],
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
      problemSize: [this.scene.rays.samples[0] * this.scene.rays.samples[1], 1, 1],
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
    pass.setBindGroup(0, this.bindGroupManager.getGroup("ray-uniforms"));
    pass.setBindGroup(1, this.bindGroupManager.getGroup("rays"));
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
    pass.setBindGroup(0, this.bindGroupManager.getGroup("ray-uniforms"));
    pass.setBindGroup(1, this.bindGroupManager.getGroup("rays"));
    pass.setBindGroup(2, this.bindGroupManager.getGroup("sort"));
    pass.setBindGroup(3, this.bindGroupManager.getGroup("collision"));
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
    pass.setBindGroup(0, this.bindGroupManager.getGroup("ray-uniforms"));
    pass.setBindGroup(1, this.bindGroupManager.getGroup("rays"));
    pass.setBindGroup(2, this.bindGroupManager.getGroup("find"));
    pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
  }

  computebitonicSort(pass: GPUComputePassEncoder) {
    const linearLayout = this.workgroups.getLayout("workgroup-per-ray");
    const totalWorkgroups = linearLayout.dispatchSize[0];

    pass.setPipeline(this.pipelineManager.get<GPUComputePipeline>("bitonic-sort"));
    pass.setBindGroup(0, this.bindGroupManager.getGroup("ray-uniforms"));
    pass.setBindGroup(1, this.bindGroupManager.getGroup("rays"));
    pass.setBindGroup(2, this.bindGroupManager.getGroup("sort"));
    pass.dispatchWorkgroups(totalWorkgroups);
  }

  updateRayWorkgroups() {
    this.workgroups.update("2d-grid-per-ray", {
      problemSize: [this.scene.rays.samples[0], this.scene.rays.samples[1], 1],
    });
    this.workgroups.update("workgroup-per-ray", {
      problemSize: [this.scene.rays.samples[0] * this.scene.rays.samples[1], 1, 1],
      strategyArgs: [QuadTree.noMaxNodesHit(this.scene.tree.depth)],
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
      constants: {
        BLOCK_SIZE: QuadTree.noMaxNodesHit(this.scene.tree.depth)
      },
      codeConstants: {
        WORKGROUP_SIZE: linearLayout.workgroupSize[0],
      }
    });
  }
}