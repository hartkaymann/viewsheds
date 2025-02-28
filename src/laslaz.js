import Promise from "bluebird"; // Assuming bluebird is needed

const pointFormatReaders = {
    0: dv => ({
        "position": [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        "intensity": dv.getUint16(12, true),
        "classification": dv.getUint8(15, true)
    }),
    1: dv => ({
        "position": [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        "intensity": dv.getUint16(12, true),
        "classification": dv.getUint8(15, true)
    }),
    2: dv => ({
        "position": [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        "intensity": dv.getUint16(12, true),
        "classification": dv.getUint8(15, true),
        "color": [dv.getUint16(20, true), dv.getUint16(22, true), dv.getUint16(24, true)]
    }),
    3: dv => ({
        "position": [dv.getInt32(0, true), dv.getInt32(4, true), dv.getInt32(8, true)],
        "intensity": dv.getUint16(12, true),
        "classification": dv.getUint8(15, true),
        "color": [dv.getUint16(28, true), dv.getUint16(30, true), dv.getUint16(32, true)]
    })
};

class LASFile {
    constructor(arraybuffer) {
        this.arraybuffer = arraybuffer;
        this.determineVersion();
        if (this.version > 13) throw new Error("Only versions <= 1.3 are supported");

        this.determineFormat();
        if (!pointFormatReaders[this.formatId]) throw new Error("Unsupported point format ID");

        this.loader = this.isCompressed ? new LAZLoader(this.arraybuffer) : new LASLoader(this.arraybuffer);
    }

    determineFormat() {
        const formatId = new Uint8Array(this.arraybuffer, 104, 1)[0];
        this.isCompressed = (formatId & 0x80) !== 0;
        this.formatId = formatId & 0x3f;
    }

    determineVersion() {
        const ver = new Int8Array(this.arraybuffer, 24, 2);
        this.version = ver[0] * 10 + ver[1];
        this.versionAsString = `${ver[0]}.${ver[1]}`;
    }

    async open() {
        return this.loader.open();
    }

    async getHeader() {
        return this.loader.getHeader();
    }

    async readData(count, start, skip) {
        return this.loader.readData(count, start, skip);
    }

    async close() {
        return this.loader.close();
    }

    getUnpacker() {
        return LASDecoder;
    }
}

class LASLoader {
    constructor(arraybuffer) {
        this.arraybuffer = arraybuffer;
    }

    open() {
        return Promise.resolve();
    }

    async getHeader() {
        return parseLASHeader(this.arraybuffer);
    }

    async readData(count, offset, skip) {
        return {
            buffer: this.arraybuffer.slice(offset, offset + count * 12),
            count,
            hasMoreData: false
        };
    }

    async close() {
        return Promise.resolve();
    }
}

class LAZLoader {
    constructor(arraybuffer) {
        this.arraybuffer = arraybuffer;
    }

    async open() {
        return Promise.resolve();
    }

    async getHeader() {
        return { pointsCount: 1000 }; // Placeholder
    }

    async readData(count, offset, skip) {
        return { buffer: new ArrayBuffer(count * 12), count };
    }

    async close() {
        return Promise.resolve();
    }
}

class LASDecoder {
    constructor(buffer, len, header) {
        this.buffer = buffer;
        this.decoder = pointFormatReaders[header.pointsFormatId];
        this.pointsCount = len;
    }

    getPoint(index) {
        return this.decoder(new DataView(this.buffer, index * 12, 12));
    }
}

function parseLASHeader(arraybuffer) {
    return {
        pointsOffset: new DataView(arraybuffer, 96, 4).getUint32(0, true),
        pointsFormatId: new DataView(arraybuffer, 100, 1).getUint8(0),
        pointsStructSize: new DataView(arraybuffer, 101, 2).getUint16(0, true),
        pointsCount: new DataView(arraybuffer, 103, 4).getUint32(0, true)
    };
}

export { LASFile };
