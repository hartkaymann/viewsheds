// Constants
const BLOCK_SIZE: u32 = 128u;
const RADIX_BITS: u32 = 4u;
const RADIX_BUCKETS: u32 = 1u << RADIX_BITS; // 16
const TOTAL_BITS: u32 = 32u;
const PASSES: u32 = TOTAL_BITS / RADIX_BITS;

// Shared workgroup memory
var<workgroup> keys: array<u32, BLOCK_SIZE>;
var<workgroup> indices: array<u32, BLOCK_SIZE>;
var<workgroup> tempKeys: array<u32, BLOCK_SIZE>;
var<workgroup> tempIndices: array<u32, BLOCK_SIZE>;
var<workgroup> histogram: array<atomic<u32>, RADIX_BUCKETS>;
var<workgroup> scanned: array<atomic<u32>, RADIX_BUCKETS>;

struct compUniforms {
    rayOrigin: vec3f,   
    startTheta: f32,        // Start horizontal angle (radians)
    endTheta: f32,          // End horizontal angle (radians)
    startPhi: f32,          // Start vertical angle (radians)
    endPhi: f32,            // End vertical angle (radians)
    raySamples: vec2<u32>,  // Grid size (X, Y)
};

struct Ray {
    origin: vec3f,
    length: f32,
    direction: vec3f,
    // 4 byte padding
};

struct QuadTreeNode {
    position: vec3f,
    childCount: u32,
    size: vec3f,
    startPointIndex: u32,
    pointCount: u32,
    startTriangleIndex: u32,
    triangleCount: u32,
};

@group(0) @binding(0) var<storage, read> nodeBuffer: array<QuadTreeNode>;
@group(0) @binding(1) var<storage, read_write> rayBuffer: array<Ray>;
@group(0) @binding(2) var<storage, read_write> rayNodeBuffer: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: compUniforms;

fn rayAABBIntersection(origin: vec3f, dir: vec3f, pos: vec3f, size: vec3f) -> f32 {
    let invDir = 1.0 / dir; // Compute inverse of ray direction

    let t1 = (pos - origin) * invDir;          // Min boundary intersection
    let t2 = (pos + size - origin) * invDir;   // Max boundary intersection

    let tMin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z)); // Entry point
    let tMax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z)); // Exit point

    // If no intersection, return -1.0
    if (tMax < max(0.0, tMin)) {
        return -1.0;
    }

    // If the ray starts inside the box, return 0.0
    return select(tMin, 0.0, tMin < 0.0);
}

fn getEntryDistance(ray: Ray, nodeIndex: u32) -> f32 {
    if (nodeIndex == 0xFFFFFFFFu) { // Invalid node index
        return 1e9; // Large distance to push it to the end
    }

    let node = nodeBuffer[nodeIndex];
    return rayAABBIntersection(ray.origin, ray.direction, node.position, node.position + node.size);
}

fn floatToSortableUint(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    return select(bits ^ 0xFFFFFFFFu, bits ^ 0x80000000u, x < 0.0);
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) gid: vec3<u32>) {    
    
    let ray = rayBuffer[gid.x];

    let local_id = lid.x;
    let block_start = gid.x * (BLOCK_SIZE + 1u);
    let count = rayNodeBuffer[block_start];
    let isActive = local_id < count;

    var key: u32 = 0u;
    var index: u32 = 0u;

    if (isActive) {
        index = rayNodeBuffer[block_start + 1u + local_id];
        let node = nodeBuffer[index];
        let distance = rayAABBIntersection(ray.origin, ray.direction, node.position, node.position + node.size);
        key = floatToSortableUint(distance);
    }

    keys[local_id] = key;
    indices[local_id] = index;

    workgroupBarrier();

    for (var curr_pass = 0u; curr_pass < PASSES; curr_pass++) {
        let shift = curr_pass * RADIX_BITS;

        // Reset histogram
        if (local_id < RADIX_BUCKETS) {
            atomicStore(&histogram[local_id], 0u);
            atomicStore(&scanned[local_id], 0u);        
        }
        workgroupBarrier();

        // Histogram step
        if (isActive) {
            let key = keys[local_id];
            let digit = (key >> shift) & (RADIX_BUCKETS - 1u);
            atomicAdd(&histogram[digit], 1u);
        }
        workgroupBarrier();

        // Exclusive scan
        if (local_id == 0u) {
            var sum = 0u;
            for (var i = 0u; i < RADIX_BUCKETS; i = i + 1u) {
                atomicStore(&scanned[i], sum);
                sum = sum + atomicLoad(&histogram[i]);
            }
        }
        workgroupBarrier();

        // Scatter
        if (isActive) {
            let key = keys[local_id];
            let idx = indices[local_id];
            let digit = (key >> shift) & (RADIX_BUCKETS - 1u);
            let offset = atomicAdd(&scanned[digit], 1u);
            tempKeys[offset] = key;
            tempIndices[offset] = idx;
        }
        workgroupBarrier();

        // Swap pointers
        if (isActive) {
            keys[local_id] = tempKeys[local_id];
            indices[local_id] = tempIndices[local_id];
        }
        workgroupBarrier();
    }

    // Write sorted indices back
    if (isActive) {
        rayNodeBuffer[block_start + 1u + local_id] = indices[local_id];
    }
}