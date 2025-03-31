// Constants
override BLOCK_SIZE: u32 = 64u;

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
    hit: u32
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

@group(3) @binding(0) var<storage, read> positionsBuffer: array<vec4f>;
@group(3) @binding(1) var<storage, read> indexBuffer: array<u32>;
@group(3) @binding(2) var<storage, read> triangleMapping: array<u32>;
@group(3) @binding(3) var<storage, read_write> pointVisibilityBuffer: array<atomic<u32>>;


fn rayIntersectsTriangle(rayPos: vec3f, rayDir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> f32 {
    let epsilon = 1e-4;

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

fn markPointHit(index: u32) {
    let wordIndex = index / 32;
    let bitIndex = index % 32;
    if (wordIndex < arrayLength(&pointVisibilityBuffer)) {
        atomicOr(&pointVisibilityBuffer[wordIndex], (1u << bitIndex));
    }
}

@compute @workgroup_size(__WORKGROUP_SIZE_X__, __WORKGROUP_SIZE_Y__, __WORKGROUP_SIZE_Z__)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rayIndex = id.x + (id.y * uniforms.raySamples.x);
    let baseOffset = rayIndex * BLOCK_SIZE;
    let leavesOffset = (1u << (2u * 8u)) / 3u; // 8 is depth of the tree

    let ray = rayBuffer[rayIndex];

    var hit = false;
    var closestDistance = 99999.0;
    var closestTriangle: vec3<u32> = vec3<u32>(0u, 1u, 2u);

    for (var i = 0u; i < rayNodeCounts[rayIndex]; i++) {
        let node = nodeBuffer[rayNodeBuffer[baseOffset + i]];
        
        for(var j = 0u; j < node.triangleCount; j++) {
            let triIndex = triangleMapping[node.startTriangleIndex + j];

            let i0 = indexBuffer[triIndex * 3 + 0];
            let i1 = indexBuffer[triIndex * 3 + 1];
            let i2 = indexBuffer[triIndex * 3 + 2];
            
            let v0 = positionsBuffer[i0].xyz;
            let v1 = positionsBuffer[i1].xyz;
            let v2 = positionsBuffer[i2].xyz;

            let t = rayIntersectsTriangle(ray.origin, ray.direction, v0, v1, v2);

            if (t > 0.0 && t < closestDistance) {
                closestDistance = t;
                closestTriangle = vec3<u32>(i0, i1, i2);
                hit = true;
            }
        }

        if (hit) {
            break;
        }
    }

    if(closestDistance < 99999.0) {
        markPointHit(closestTriangle.x);
        markPointHit(closestTriangle.y);
        markPointHit(closestTriangle.z);
        
        rayBuffer[rayIndex].length = closestDistance;
        rayBuffer[rayIndex].hit = select(0u, 1u, hit);
    }
}