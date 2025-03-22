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

//@group(0) @binding(0) var<storage, read> positionsBuffer: array<vec4f>;
//@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
// @group(0) @binding(6) var<storage, read_write> closestHitBuffer: array<atomic<u32>, 4>;
// @group(0) @binding(3) var<storage, read> triangleMapping: array<u32>;
// @group(0) @binding(4) var<storage, read_write> pointVisibilityBuffer: array<atomic<u32>>;
@group(0) @binding(0) var<storage, read> nodeBuffer: array<QuadTreeNode>;
@group(0) @binding(1) var<storage, read_write> rayBuffer: array<Ray>;
@group(0) @binding(2) var<storage, read_write> rayNodeBuffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> rayNodeCounts: array<u32>;
@group(0) @binding(4) var<storage, read_write> nodeVisibilityBuffer: array<atomic<u32>>;

@group(0) @binding(5) var<uniform> uniforms: compUniforms;

fn rayAABBIntersection(origin: vec3f, dir: vec3f, pos: vec3f, size: vec3f) -> bool {
    let invDir = 1.0 / dir; // Compute inverse of ray direction

    let t1 = (pos - origin) * invDir;          // Min boundary intersection
    let t2 = (pos + size - origin) * invDir;   // Max boundary intersection

    let tMin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z)); // Entry point
    let tMax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z)); // Exit point

    // Return true if the ray intersects the box, otherwise false
    return !(tMax < max(0.0, tMin));
}

fn rayIntersectsTriangle(rayPos: vec3f, rayDir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> f32 {
    let epsilon = 0.000001;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);

    // Check ray parallel to triangle
    if (abs(a) < epsilon) {
        return -1.0;
    }

    let f = 1.0 / a;
    let s = rayPos - v0;
    let u = f * dot(s, h);

    // Check intersection outside triangle
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }

    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);

    if (v < 0.0 || (u + v) > 1.0) {
        return -1.0;
    }

    let t = f * dot(edge2, q);

    if (t > epsilon) {
        return t;
    }

    return -1.0;
}

// fn markPointHit(index: u32) {
//     let wordIndex = index / 32;
//     let bitIndex = index % 32;
//     atomicOr(&pointVisibilityBuffer[wordIndex], (1u << bitIndex));
// }

fn markNodeHit(index: u32) {
    let wordIndex = index / 32;
    let bitIndex = index % 32;
    atomicOr(&nodeVisibilityBuffer[wordIndex], (1u << bitIndex));
}

@compute @workgroup_size(8, 8, 1) // 8x8 rays
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let depth = 6u;
    let rayIndex = id.x + (id.y * uniforms.raySamples.x);
    let baseOffset = rayIndex * 128;
    let leafOffset = (1u << (2u * depth)) / 3u; // bit-shift instead of pow

    let gridSize = vec2f(f32(uniforms.raySamples.x), f32(uniforms.raySamples.y));
    let id2D = vec2f(f32(id.x) / gridSize.x, f32(id.y) / gridSize.y); 
    
    let theta = uniforms.startTheta + id2D.x * (uniforms.endTheta - uniforms.startTheta);
    let phi = acos(mix(cos(uniforms.startPhi), cos(uniforms.endPhi), id2D.y));
    
    let rayPos = uniforms.rayOrigin;
    let rayDir = normalize(vec3f(
        cos(theta) * sin(phi), // X
        cos(phi),              // Y
        sin(theta) * sin(phi)  // Z
    ));
    let ray = Ray(rayPos, 0.0, rayDir);

    // Stack-based iterative traversal to collect leaf nodes
    var stackPointer: i32 = 0;
    // Maximum stack size depends on depth of the tree:
    // S = 2^(D+2)-2
    var stack: array<u32, 256>; // For depth 6
    stack[0] = 0; // Start with the root node

    var leafCount: u32 = 0u;

    while (stackPointer >= 0) {
        let nodeIndex = stack[stackPointer];
        stackPointer -= 1; // Pop from stack

        let node = nodeBuffer[nodeIndex];
        
        // If the ray doesn't intersect this node, skip it
        if (!rayAABBIntersection(ray.origin, ray.direction, node.position, node.size)) {
            continue;
        }

        // If it's a leaf node, add it to the list
        if (node.childCount == 0u) {
            rayNodeBuffer[baseOffset + leafCount] = nodeIndex;
            leafCount += 1;

            markNodeHit(nodeIndex - leafOffset);
            continue;
        }

        // If it's an internal node, push all 4 children onto the stack
        let firstChildIndex = 4u * nodeIndex + 1u;                                       
        for (var i = 0u; i < 4u; i++) {
            stackPointer += 1;
            stack[stackPointer] = firstChildIndex + i;
        }
    }
    rayNodeCounts[rayIndex] = leafCount;

    // Store ray in the buffer
    let linearIndex = id.x + (id.y * uniforms.raySamples.x);
    if (linearIndex < arrayLength(&rayBuffer)) {
        rayBuffer[linearIndex] = Ray(ray.origin, 100.0, ray.direction);
    }

    // let threadIndex = id.z;

    // var hit = false;
    // var closestDistance = 99999.0;
    // var closestTriangle: vec3<u32> = vec3<u32>(0u, 0u, 0u); // Store indices of closest triangle

    // for (var i = 0u; i < leafCount; i++) {
    //     let nodeIndex = leafNodes[i];
    //     let node = nodeBuffer[nodeIndex];
        
    //     for (var j = threadIndex; j < node.triangleCount; j += 4u) {
    //         let triIndex = triangleMapping[node.startTriangleIndex + j];

    //         let i0 = indexBuffer[triIndex * 3 + 0];
    //         let i1 = indexBuffer[triIndex * 3 + 1];
    //         let i2 = indexBuffer[triIndex * 3 + 2];
            
    //         let v0 = positionsBuffer[i0].xyz;
    //         let v1 = positionsBuffer[i1].xyz;
    //         let v2 = positionsBuffer[i2].xyz;

    //         let t = rayIntersectsTriangle(rayPos, rayDir, v0, v1, v2);

    //         if (t > 0.0) {
    //             let prevDist = bitcast<f32>(atomicLoad(&closestHitBuffer[0]));

    //             if (t < prevDist) {
    //                 atomicMin(&closestHitBuffer[0], bitcast<u32>(t));
    //     	        atomicStore(&closestHitBuffer[1], i0);
    //     	        atomicStore(&closestHitBuffer[2], i1);
    //     	        atomicStore(&closestHitBuffer[3], i2);
    //                 closestDistance = t;
    //             }

    //             hit = true;
    //         }
    //     }

    //     if (hit) {
    //         break;
    //     }
    // }

    // workgroupBarrier(); 
    // if (bitcast<f32>(atomicLoad(&closestHitBuffer[0])) == closestDistance) {
    //     let tri0 = atomicLoad(&closestHitBuffer[1]);
    //     let tri1 = atomicLoad(&closestHitBuffer[2]);
    //     let tri2 = atomicLoad(&closestHitBuffer[3]);

    //     if(closestDistance < 99999.0) {

    //         markPointHit(tri0);
    //         markPointHit(tri1);
    //         markPointHit(tri2);
    //     }


}