import { BufferManager } from './BufferManager';
import { QuadTree } from './Optimization';
import { Profiler } from './Profiler';
import { RayDataTable } from './ui/RayDataTable';

export class Utils {

    static async copyAndDisplayRayDebugData(
        device: GPUDevice,
        bufferManager: BufferManager,
        raySamples: [number, number],
        treeDepth: number
    ) {
        const rayCount = raySamples[0] * raySamples[1];
        const blockSize = QuadTree.noMaxNodesHit(treeDepth);
        const bufferSize = rayCount * blockSize * 4;

        const srcDistanceBuffer = bufferManager.get("debug_distance");
        const srcIndexBuffer = bufferManager.get("ray_nodes");
        const dstDistanceBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const dstIndexBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });


        const encoder = device.createCommandEncoder();

        encoder.copyBufferToBuffer(srcDistanceBuffer, 0, dstDistanceBuffer, 0, bufferSize);
        encoder.copyBufferToBuffer(srcIndexBuffer, 0, dstIndexBuffer, 0, bufferSize);

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone(); // Ensure copy is done before mapping

        if (dstDistanceBuffer.mapState !== "unmapped") dstDistanceBuffer.unmap();
        if (dstIndexBuffer.mapState !== "unmapped") dstIndexBuffer.unmap();

        await Promise.all([
            dstDistanceBuffer.mapAsync(GPUMapMode.READ),
            dstIndexBuffer.mapAsync(GPUMapMode.READ),
        ]);

        const distanceArray = new Float32Array(dstDistanceBuffer.getMappedRange().slice(0));
        const indexArray = new Uint32Array(dstIndexBuffer.getMappedRange().slice(0));

        Utils.displayRayData(indexArray, distanceArray, rayCount, blockSize);

        dstDistanceBuffer.unmap();
        dstIndexBuffer.unmap();

        dstDistanceBuffer.destroy();
        dstIndexBuffer.destroy();
    }

    static displayRayData(indexArray: Uint32Array, distanceArray: Float32Array, rayCount: number, blockSize: number) {
        const rayDataTable = new RayDataTable({
            containerId: "debug-table-container",
            rayCount,
            blockSize,
            indexArray,
            distanceArray
        });
        rayDataTable.displayRayData();
    }

    static showToast(message: string, type: 'error' | 'warn' = 'error') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        const container = document.getElementById('toast-container');
        container?.appendChild(toast);

        setTimeout(() => toast.remove(), 5000);
    }
}