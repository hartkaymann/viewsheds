// Constants
const BLOCK_SIZE: u32 = 128u;

// Shared workgroup memory
var<workgroup> keys: array<f32, BLOCK_SIZE>;
var<workgroup> indices: array<u32, BLOCK_SIZE>;


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
@group(0) @binding(1) var<storage, read> rayBuffer: array<Ray>;
@group(0) @binding(2) var<storage, read> rayNodeCounts: array<u32>;
@group(0) @binding(3) var<storage, read_write> rayNodeBuffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> debugDistances: array<f32>;

@group(0) @binding(5) var<uniform> uniforms: compUniforms;

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
    if (nodeIndex == 0xFFFFFFFFu) {
        return 1e9;
    }

    let node = nodeBuffer[nodeIndex];
    return rayAABBIntersection(ray.origin, ray.direction, node.position, node.size);
}

fn bitonicCompare(i: u32, j: u32, dir: bool) {
    if (j >= BLOCK_SIZE) { return; }

    let key_i = keys[i];
    let key_j = keys[j];

    let idx_i = indices[i];
    let idx_j = indices[j];

    let swap = (key_i > key_j) == dir;

    if (swap) {
        keys[i] = key_j;
        keys[j] = key_i;
        indices[i] = idx_j;
        indices[j] = idx_i;
    }
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) gid: vec3<u32>) {    
    
    let local_id = lid.x;
    let block_start = gid.x * BLOCK_SIZE;
    let ray = rayBuffer[gid.x];
    let count = min(rayNodeCounts[gid.x], BLOCK_SIZE);
    let isActive = local_id < count;

    var key: u32 = 0u;
    var index: u32 = 0u;

    // Load keys and indices
    if (isActive) {
        let nodeIndex = rayNodeBuffer[block_start + local_id];
        indices[local_id] = nodeIndex;
        keys[local_id] = getEntryDistance(ray, nodeIndex);
    } else {
        // Fill with max values to push them to the end
        indices[local_id] = 0xFFFFFFFFu;
        keys[local_id] = 1e9;
    }

    workgroupBarrier();

    // Bitonic sort in shared memory
    var k = 2u;
    while (k <= BLOCK_SIZE) {
        var j = k >> 1u;
        while (j > 0u) {
            let ixj = local_id ^ j;
            if (ixj > local_id) {
                let ascending = ((local_id & k) == 0u);
                bitonicCompare(local_id, ixj, ascending);
            }
            workgroupBarrier();
            j = j >> 1u;
        }
        k = k << 1u;
    }

    workgroupBarrier();

    // Write back sorted indices
    if (isActive) {
        rayNodeBuffer[block_start + local_id] = indices[local_id];

        let dist = getEntryDistance(ray, indices[local_id]);
        debugDistances[block_start + local_id] = dist;
    }
}