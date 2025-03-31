override TREE_DEPTH = 6u;
override BLOCK_SIZE = 128u;

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

//@group(0) @binding(0) var<storage, read> positionsBuffer: array<vec4f>;
//@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
// @group(0) @binding(6) var<storage, read_write> closestHitBuffer: array<atomic<u32>, 4>;
// @group(0) @binding(3) var<storage, read> triangleMapping: array<u32>;
// @group(0) @binding(4) var<storage, read_write> pointVisibilityBuffer: array<atomic<u32>>;
@group(0) @binding(0) var<uniform> uniforms: compUniforms;

@group(1) @binding(0) var<storage, read_write> rayBuffer: array<Ray>;

@group(2) @binding(0) var<storage, read> nodeBuffer: array<QuadTreeNode>;
@group(2) @binding(1) var<storage, read_write> rayNodeCounts: array<u32>;
@group(2) @binding(2) var<storage, read_write> rayNodeBuffer: array<u32>;
@group(2) @binding(3) var<storage, read_write> nodeVisibilityBuffer: array<atomic<u32>>;


fn rayAABBIntersection(origin: vec3f, dir: vec3f, pos: vec3f, size: vec3f) -> bool {
    let invDir = 1.0 / dir; // Compute inverse of ray direction

    let t1 = (pos - origin) * invDir;          // Min boundary intersection
    let t2 = (pos + size - origin) * invDir;   // Max boundary intersection

    let tMin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z)); // Entry point
    let tMax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z)); // Exit point

    // Return true if the ray intersects the box, otherwise false
    return !(tMax < max(0.0, tMin));
}

fn markNodeHit(index: u32) {
    let wordIndex = index / 32;
    let bitIndex = index % 32;
    atomicOr(&nodeVisibilityBuffer[wordIndex], (1u << bitIndex));
}

@compute @workgroup_size(__WORKGROUP_SIZE_X__, __WORKGROUP_SIZE_Y__, __WORKGROUP_SIZE_Z__)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rayIndex = id.x + (id.y * uniforms.raySamples.x);
    let baseOffset = rayIndex * BLOCK_SIZE;
    let leafOffset = (1u << (2u * TREE_DEPTH)) / 3u; // bit-shift instead of pow

    let ray = rayBuffer[rayIndex];

    var stackPointer: i32 = 0;
    var stack: array<u32, __MAX_STACK_SIZE__>; 
    stack[0] = 0;

    var leafCount: u32 = 0u;

    while (stackPointer >= 0) {
        let nodeIndex = stack[stackPointer];
        stackPointer -= 1; // Pop from stack

        let node = nodeBuffer[nodeIndex];
        
        // No intersection, skip
        if (!rayAABBIntersection(ray.origin, ray.direction, node.position, node.size)) {
            continue;
        }

        // Add leaf node to list
        if (node.childCount == 0u) {
            rayNodeBuffer[baseOffset + leafCount] = nodeIndex;
            leafCount += 1;

            rayBuffer[rayIndex].hit = 1u;
            markNodeHit(nodeIndex - leafOffset);
            continue;
        }

        // For internal node, add 4 children
        let firstChildIndex = 4u * nodeIndex + 1u;                                       
        for (var i = 0u; i < 4u; i++) {
            if (stackPointer + 1 < __MAX_STACK_SIZE__) {
                stackPointer += 1;
                stack[stackPointer] = firstChildIndex + i;
            }
        }
    }
    rayNodeCounts[rayIndex] = leafCount;
}