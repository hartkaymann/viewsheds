import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { QuadTree } from "./Optimization";
import { PipelineManager } from "./PipelineManager";
import { Scene } from "./Scene";

export class SceneSyncer {

  scene: Scene;
  device: GPUDevice;
  bufferManager: BufferManager;
  bindGroupManager: BindGroupManager;


  constructor(scene: Scene, device: GPUDevice, bufferManager: BufferManager, bindGroupManager: BindGroupManager) {
    this.scene = scene;
    this.device = device;
    this.bufferManager = bufferManager;
    this.bindGroupManager = bindGroupManager;
  }

  async setPointData() {
    this.bufferManager.resize("points", this.scene.points.byteLength);
    this.bufferManager.write("points", this.scene.points);

    this.bufferManager.resize("colors", this.scene.colors.byteLength);
    this.bufferManager.write("colors", this.scene.colors);

    this.bufferManager.resize("classification", this.scene.classification.byteLength);
    this.bufferManager.write("classification", this.scene.classification);

    this.bufferManager.resize("point_visibility", Math.ceil(this.scene.points.length / 3 / 32) * Uint32Array.BYTES_PER_ELEMENT);
  }

  async setNodeData(resize: boolean = true) {
    if (resize) {
      const rayCount = this.scene.rays.samples[0] * this.scene.rays.samples[1];
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

  }

  async setMeshData() {
    this.bufferManager.resize("indices", this.scene.indices.byteLength);
    this.bufferManager.write("indices", this.scene.indices);

    this.bufferManager.resize("node_to_triangle", this.scene.nodeToTriangles.byteLength);
    this.bufferManager.write("node_to_triangle", this.scene.nodeToTriangles);

  }

  updateRaySamples(samples: [number, number] = [1, 1]) {
    this.scene.rays.samples[0] = samples[0];
    this.scene.rays.samples[1] = samples[1];

    this.device.queue.writeBuffer(this.bufferManager.get("ray_uniforms"), 32, new Uint32Array([
      this.scene.rays.samples[0], this.scene.rays.samples[1],
    ]));
  }

  updateThetaPhi(theta: [number, number] = [0, 0], phi: [number, number] = [0, 0]) {
    this.device.queue.writeBuffer(
      this.bufferManager.get("ray_uniforms"), 12,
      new Float32Array([theta[0], theta[1], phi[0], phi[1]]));
  }

  updateRayOrigin(origin: [number, number, number] = [0, 0, 0]) {
    this.device.queue.writeBuffer(this.bufferManager.get("ray_uniforms"), 0, new Float32Array(origin));
  }

  resizeRayRelatedBuffers() {
    const rayCount = this.scene.rays.samples[0] * this.scene.rays.samples[1];
    const maxNodesHit = QuadTree.noMaxNodesHit(this.scene.tree.depth);

    this.bufferManager.resize("rays", rayCount * 2 * 4 * 4);
    this.bufferManager.resize("ray_nodes", rayCount * maxNodesHit * 4);
    this.bufferManager.resize("debug_distance", rayCount * maxNodesHit * 4);
    this.bufferManager.resize("ray_node_counts", rayCount * 4);
    this.bufferManager.resize("ray_node_counts", rayCount * 4);
    this.bufferManager.resize("copy_distances_buffer", rayCount * maxNodesHit * 4);
    this.bufferManager.resize("copy_indices_buffer", rayCount * maxNodesHit * 4);
  }
}