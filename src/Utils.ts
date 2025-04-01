import { Profiler } from './Profiler';
import { RayDataTable } from './ui/RayDataTable';

export class Utils {
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
}