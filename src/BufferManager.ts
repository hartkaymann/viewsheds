
interface BufferInitConfig {
    name: string;
    size: number;
    usage: GPUBufferUsageFlags;
    label?: string;
    data?: ArrayBufferView | ArrayBuffer;
}

export class BufferManager {

    device: GPUDevice;
    buffers: Map<string, GPUBuffer>;
    readonly maxSize: number;

    constructor(device: GPUDevice) {
        this.device = device;
        this.buffers = new Map();
        this.maxSize = device.limits.maxStorageBufferBindingSize;

    }

    initBuffers(configs: BufferInitConfig[]): void {
        for (const { name, size, usage, label, data } of configs) {
            const clampedSize = Math.max(Math.min(size, this.device.limits.maxStorageBufferBindingSize), 4);
            const buffer = this.device.createBuffer({
                label: label ?? `buffer-${name}`,
                size: clampedSize,
                usage,
            });
            this.buffers.set(name, buffer);

            if (data) {
                this.write(name, data);
            }
        }
    }

    createBuffer(name: string, requestedSize: number, usage: GPUBufferUsageFlags): GPUBuffer {
        const size = Math.min(requestedSize, this.maxSize);
        const buffer = this.device.createBuffer({
            label: `buffer-${name}`,
            size,
            usage,
        });
        this.buffers.set(name, buffer);
        return buffer;
    }

    write(
        name: string,
        data: ArrayBuffer | ArrayBufferView,
        offset: number = 0,
        dataOffset: number = 0,
        size?: number
    ): void {
        const buffer = this.buffers.get(name);
        if (!buffer) return;

        if (data instanceof ArrayBuffer) {
            this.writeFromArrayBuffer(buffer, name, data, offset, dataOffset, size);
        } else {
            this.writeFromArrayBufferView(buffer, name, data, offset, dataOffset, size);
        }
    }

    private writeFromArrayBuffer(
        buffer: GPUBuffer,
        name: string,
        data: ArrayBuffer,
        offset: number,
        dataOffset: number,
        size?: number
    ): void {
        const bufferByteLength = buffer.size;
        const maxWritableSize = bufferByteLength - offset;
        const writeSize = size ?? data.byteLength - dataOffset;
        const safeSize = Math.min(writeSize, maxWritableSize);

        if (safeSize <= 0) {
            console.warn(`[BufferManager] Write to '${name}' skipped: size exceeds buffer limit.`);
            return;
        }

        this.device.queue.writeBuffer(buffer, offset, data, dataOffset, safeSize);
    }

    private writeFromArrayBufferView(
        buffer: GPUBuffer,
        name: string,
        data: ArrayBufferView,
        offset: number,
        dataOffset: number,
        size?: number
    ): void {
        const bytesPerElement = (data as any).BYTES_PER_ELEMENT ?? 1;
        const bufferByteLength = buffer.size;
        const maxWritableSize = bufferByteLength - offset;
        const writeSizeBytes = size ?? (data.byteLength - dataOffset * bytesPerElement);
        const safeSize = Math.min(writeSizeBytes, maxWritableSize);

        if (safeSize <= 0) {
            console.warn(`[BufferManager] Write to '${name}' skipped: size exceeds buffer limit.`);
            return;
        }

        this.device.queue.writeBuffer(
            buffer,
            offset,
            data,
            dataOffset,
            safeSize / bytesPerElement
        );
    }

    get(name: string): GPUBuffer | undefined {
        return this.buffers.get(name);
    }

    resize(name: string, newSize: number): GPUBuffer | undefined {
        const oldBuffer = this.buffers.get(name);
        if (!oldBuffer) return;

        const clampedSize = Math.min(newSize, this.device.limits.maxStorageBufferBindingSize);

        const newBuffer = this.device.createBuffer({
            label: oldBuffer.label,
            size: clampedSize,
            usage: oldBuffer.usage,
        });

        this.buffers.set(name, newBuffer);
        return newBuffer;
    }

    clear(name: string): void {
        const buffer = this.buffers.get(name);
        if (!buffer) return;

        const zeroData = new Uint32Array(buffer.size / 4);
        this.device.queue.writeBuffer(buffer, 0, zeroData);
    }

    destroy(name: string): void {
        const buffer = this.buffers.get(name);
        if (!buffer) return;

        buffer.destroy();              // Release GPU memory
        this.buffers.delete(name);     // Remove reference from the map
    }
}