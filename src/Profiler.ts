import { BufferManager } from "./BufferManager";

export class Profiler {

    device: GPUDevice;
    bufferManager: BufferManager | null = null;

    gpuMemoryMax: number = 0;
    gpuMemoryUsage: number = 0;

    constructor(device: GPUDevice) {
        this.device = device;
        this.gpuMemoryMax = device.limits.maxStorageBufferBindingSize;
    }

    setBufferManager(manager: BufferManager) {
        this.bufferManager = manager;
    
        manager.onResize((name, newSize) => {
            this.updatePanel();
        });
    }

    getTotalBufferSize(): number {
        let totalSize = 0;
        this.bufferManager?.buffers.forEach(buffer => {
            totalSize += buffer.size;
        });
        return totalSize;
    }

    getBuffersSortedBySize(): { name: string; size: number }[] {
        if (!this.bufferManager) return [];
    
        const sorted = [...this.bufferManager.buffers.entries()]
            .map(([name, tracked]) => ({
                name,
                size: tracked.size ?? 0,
            }))
            .sort((a, b) => b.size - a.size);
    
        return sorted;
    }

    logBufferSizes() {
        const list = this.getBuffersSortedBySize();
        console.log('--- GPU Buffers by size ---');
        list.forEach(buf => {
            console.log(`${buf.name}: ${(buf.size / 1024 / 1024).toFixed(2)} MB`);
        });
    }

    updatePanel() {
        const totalSizeMB = this.getTotalBufferSize() / 1024 / 1024;
        const gpuMemEl = document.getElementById("gpu-mem")!;
        const listEl = document.getElementById("buffer-list")!;
        const toggleEl = document.getElementById("buffer-toggle")!;

        gpuMemEl.textContent = `GPU: ${totalSizeMB.toFixed(2)} MB`;

        const buffers = this.getBuffersSortedBySize();
        listEl.innerHTML = buffers.map(buf => {
            const sizeMB = (buf.size / 1024 / 1024).toFixed(2);
            return `<div class="buffer-row"><span>${buf.name}</span><span>${sizeMB} MB</span></div>`;
        }).join("");

        // Toggle logic
        if (!toggleEl.dataset.initialized) {
            let isExpanded = false;
            toggleEl.onclick = () => {
                isExpanded = !isExpanded;
                listEl.style.display = isExpanded ? "block" : "none";
                toggleEl.textContent = isExpanded ? "▲ Hide Buffers" : "▼ Show Buffers";
            };
            toggleEl.dataset.initialized = "true";
        }
    }
} 