// Constants
override BLOCK_SIZE: u32 = 64u;

// Shared workgroup memory
var<workgroup> keys: array<f32, __WORKGROUP_SIZE__>;
var<workgroup> indices: array<u32, __WORKGROUP_SIZE__>;

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
    hit: u32,
};

struct QuadTreeNode {
    position: vec3f,
    childCount: u32,
    size: vec3f,
    startPointIndex: u32,
    pointCount: u32,
    startTriangleIndex: u32,
    triangleCount: u32,
    isFirst: u32,
};

@group(0) @binding(0) var<uniform> uniforms: compUniforms;

@group(1) @binding(0) var<storage, read_write> rayBuffer: array<Ray>;

@group(2) @binding(0) var<storage, read_write> nodeBuffer: array<QuadTreeNode>;
@group(2) @binding(1) var<storage, read> rayNodeCounts: array<u32>;
@group(2) @binding(2) var<storage, read_write> rayNodeBuffer: array<u32>;
@group(2) @binding(3) var<storage, read_write> debugDistances: array<f32>;


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
    if (i >= __WORKGROUP_SIZE__ || j >= __WORKGROUP_SIZE__) { return; }

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

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) gid: vec3<u32>) {    
    
    let threadsPerBlock = BLOCK_SIZE;
    let blocksPerGroup = __WORKGROUP_SIZE__ / threadsPerBlock;

    let blockIndexInGroup = lid.x / BLOCK_SIZE;
    let indexInBlock = lid.x % BLOCK_SIZE;
    let globalBlockIndex = gid.x * blocksPerGroup + blockIndexInGroup;

    let sharedIndex = blockIndexInGroup * BLOCK_SIZE + indexInBlock;

    let blockStart = globalBlockIndex * BLOCK_SIZE;
    let ray = rayBuffer[globalBlockIndex];
    let count = min(rayNodeCounts[globalBlockIndex], BLOCK_SIZE);

    let isActive = indexInBlock < count;

    var key: u32 = 0u;
    var index: u32 = 0u;

    // Load keys and indices
    let nodeIndex = select(0xFFFFFFFFu, rayNodeBuffer[blockStart + indexInBlock], isActive);
    indices[sharedIndex] = nodeIndex;
    keys[sharedIndex] = getEntryDistance(ray, nodeIndex);

    workgroupBarrier();

    // Bitonic sort in shared memory
    var k = 2u;
    while (k <= BLOCK_SIZE) {
        var j = k >> 1u;
        while (j > 0u) {
            let ixj = indexInBlock ^ j;
            if (ixj > indexInBlock) {
                let ixjShared = blockIndexInGroup * BLOCK_SIZE + ixj;
                let ascending = ((indexInBlock & k) == 0u);
                bitonicCompare(sharedIndex, ixjShared, ascending);
            }
            workgroupBarrier();
            j = j >> 1u;
        }
        k = k << 1u;
    }

    workgroupBarrier();

    // Write back sorted indices
    if (isActive) {
        let sortedNode = indices[sharedIndex];
        rayNodeBuffer[blockStart + indexInBlock] = sortedNode;
        debugDistances[blockStart + indexInBlock] = getEntryDistance(ray, sortedNode);
    }

    if (indexInBlock == 0u && count > 0u) {
        let closestNode = rayNodeBuffer[blockStart];
        let closestDistance = getEntryDistance(ray, closestNode);
    
        rayBuffer[globalBlockIndex].length = closestDistance;
        nodeBuffer[closestNode].isFirst = 1u;
    }
}