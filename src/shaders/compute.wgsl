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
    // Automatic padding 4 bytes here
};

struct QuadTreeNode {
    position: vec3f,
    size: vec3f,
    startPointIndex: u32,
    pointCount: u32,
    startTriangleIndex: u32,
    triangleCount: u32,
};

@group(0) @binding(0) var<storage, read> positionsBuffer: array<vec4f>;
@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
@group(0) @binding(2) var<storage, read> quadtree: array<QuadTreeNode>;
@group(0) @binding(3) var<storage, read> triangleMapping: array<u32>;

@group(0) @binding(4) var<storage, read_write> visibilityBuffer: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> rayBuffer: array<Ray>;

@group(0) @binding(6) var<uniform> uniforms: compUniforms;


fn rayIntersectsAABB(origin: vec3f, dir: vec3f, pos: vec3f, size: vec3f) -> bool {
    let invDir = 1.0 / dir; // Compute inverse of ray direction

    let t1 = (pos - origin) * invDir; // Intersection distances to min boundaries
    let t2 = (pos + size - origin) * invDir; // Intersection distances to max boundaries

    let tMin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tMax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    return tMax >= max(0.0, tMin); // True if ray enters before exiting
}

fn getRayAABBEntryDistance(origin: vec3f, direction: vec3f, aabbMin: vec3f, aabbMax: vec3f) -> f32 {
    let invDir = 1.0 / direction; // Compute inverse direction

    let tMin = (aabbMin - origin) * invDir;
    let tMax = (aabbMax - origin) * invDir;

    let tEnter = max(max(min(tMin.x, tMax.x), min(tMin.y, tMax.y)), min(tMin.z, tMax.z));
    let tExit = min(min(max(tMin.x, tMax.x), max(tMin.y, tMax.y)), max(tMin.z, tMax.z));

    // If the ray starts inside the box, entry distance is 0
    return select(tEnter, 0.0, tEnter < 0.0 || tEnter > tExit);
}

fn getEntryDistance(ray: Ray, nodeIndex: u32) -> f32 {
    let node = quadtree[nodeIndex];
    return getRayAABBEntryDistance(ray.origin, ray.direction, node.min, node.min + node.max);
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

fn markHit(index: u32) {
    let wordIndex = index / 32;
    let bitIndex = index % 32;
    atomicOr(&visibilityBuffer[wordIndex], (1u << bitIndex));
}

@compute @workgroup_size(8, 8, 4) // 8x8 rays, 4 threads per ray testing different points
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rayOrigin = uniforms.rayOrigin;
    let maxSteps = uniforms.maxSteps;
    let stepSize = uniforms.stepSize;

    let gridSize = vec2f(f32(uniforms.raySamples.x), f32(uniforms.raySamples.y));
    let id2D = vec2f(f32(id.x) / gridSize.x, f32(id.y) / gridSize.y); 
    
    let theta = uniforms.startTheta + id2D.x * (uniforms.endTheta - uniforms.startTheta);
    let phi = acos(mix(cos(uniforms.startPhi), cos(uniforms.endPhi), id2D.y));

    let rayDir = normalize(vec3f(
        cos(theta) * sin(phi), // X
        cos(phi),              // Y
        sin(theta) * sin(phi)  // Z
    ));

    var rayPos = rayOrigin;

    let threadIndex = id.z;

    // Stack-based iterative traversal to collect leaf nodes
    var stackPointer: i32 = 0;
    var stack: array<u32, 32>; // Enough space for worst-case depth

    stack[0] = 0; // Start with the root node
    var leafNodes: array<u32, 64>; // Buffer for intersected leaf nodes
    var leafCount: u32 = 0u;

    while (stackPointer >= 0) {
        let nodeIndex = stack[stackPointer];
        stackPointer -= 1u; // Pop from stack

        let node = quadtree[nodeIndex];
        
        // If the ray doesn't intersect this node, skip it
        if (!rayIntersectsAABB(ray.origin, ray.direction, node.position, node.size)) {
            continue;
        }

        // If it's a leaf node, add it to the list
        if (node.pointCount > 0u) {
            leafNodes[leafCount] = nodeIndex;
            leafCount += 1u;
            continue;
        }

        // If it's an internal node, push all 4 children onto the stack
        let firstChildIndex = 4u * nodeIndex + 1u;                                       
        for (var i = 0u; i < 4u; i++) {
            stackPointer += 1u;
            stack[stackPointer] = firstChildIndex + i;
        }
    }

    // Simple selection sort for small lists
    for (var i = 0u; i < leafCount - 1u; i++) {
        var minIndex = i;
        for (var j = i + 1u; j < leafCount; j++) {
            if (getEntryDistance(ray, leafNodes[j]) < getEntryDistance(ray, leafNodes[minIndex])) {
                minIndex = j;
            }
        }

        // Swap elements
        let temp = leafNodes[i];
        leafNodes[i] = leafNodes[minIndex];
        leafNodes[minIndex] = temp;
    }


    var hit = false;
    var closestDistance = 99999.0;
    var closestTriangle: vec3<u32> = vec3<u32>(0u, 0u, 0u); // Store indices of closest triangle

    for (var i = 0u; i < leafCount; i++) {
        let nodeIndex = leafNodes[i];
        let node = nodeBuffer[nodeIndex];
        
        for (var j = 0u; j < node.triangleCount; j++) {
            let triIndex = triangleMapping[node.startTriangleIndex + j];

            i0 = indexBuffer[triIndex * 3 + 0];
            i1 = indexBuffer[triIndex * 3 + 1];
            i2 = indexBuffer[triIndex * 3 + 2];
            
            let v0 = positionsBuffer[i0].xyz;
            let v1 = positionsBuffer[i1].xyz;
            let v2 = positionsBuffer[i2].xyz;

            let t = rayIntersectsTriangle(rayPos, rayDir, v0, v1, v2);

            if (t > 0.0 && t < closestDistance) {
                hit = true;
                closestDistance = t;
                closestTriangle = vec3<u32>(i0, i1, i2);
            }
        }

        if (hit) {
            break;
        }
    }

    if(closestDistance < 99999.0) {
        markHit(closestTriangle.x);
        markHit(closestTriangle.y);
        markHit(closestTriangle.z);
    }

    // Store ray in the buffer
    let linearIndex = id.x + (id.y * uniforms.raySamples.x);
    if (linearIndex < arrayLength(&rayBuffer)) {
        rayBuffer[linearIndex] = Ray(rayOrigin, closestDistance, rayDir);
    }
}