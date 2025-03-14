struct compUniforms {
    rayOrigin: vec3f,   
    stepSize: f32,          // Step size
    startTheta: f32,        // Start horizontal angle (radians)
    endTheta: f32,          // End horizontal angle (radians)
    startPhi: f32,          // Start vertical angle (radians)
    endPhi: f32,            // End vertical angle (radians)
    raySamples: vec2<u32>,   // Grid size (X, Y)
    maxSteps: u32,         // Maximum number of steps
};

struct Ray {
    origin: vec3f,
    steps: u32,
    direction: vec3f,
    stepSize: f32
};

@group(0) @binding(0) var<storage, read> positionsBuffer: array<vec4f>;
@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> visibilityBuffer: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> uniforms: compUniforms;
@group(0) @binding(5) var<storage, read_write> rayBuffer: array<Ray>;

fn rayStepIntersectsTriangle(rayStepPos: vec3<f32>, rayDir: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);

    if (abs(a) < 0.00001) { return false; } // Ray parallel to triangleecision

    let f = 1.0 / a;
    let s = rayStepPos - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return false; }

    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);
    if (v < 0.0 || (u + v) > 1.0) { return false; }

    let t = f * dot(edge2, q);
    
    return t > 0.00001 && t < 1.0;// Adjusted threshold for precision
}

fn markHit(index: u32) {
    let wordIndex = index / 32;
    let bitIndex = index % 32;
    atomicOr(&visibilityBuffer[wordIndex], (1u << bitIndex));
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rayOrigin = uniforms.rayOrigin;
    let maxSteps = uniforms.maxSteps;
    let stepSize = uniforms.stepSize;

    let gridSize = vec2f(f32(uniforms.raySamples.x), f32(uniforms.raySamples.y));
    let id2D = vec2<f32>(f32(id.x) / gridSize.x, f32(id.y) / gridSize.y); 
    
    let theta = uniforms.startTheta + id2D.x * (uniforms.endTheta - uniforms.startTheta);
    let phi = acos(mix(cos(uniforms.startPhi), cos(uniforms.endPhi), id2D.y));

    let rayDir = normalize(vec3f(
        cos(theta) * sin(phi), // X
        cos(phi),              // Y
        sin(theta) * sin(phi)  // Z
    ));

    var rayPos = rayOrigin;
    var stepsTaken = 0u;
    var hit = false;

    for (var i = 0u; i < maxSteps; i++) {
        rayPos += rayDir * stepSize; // Move the ray forward
        stepsTaken = i + 1u;

        for (var j = 0u; j <  arrayLength(&indexBuffer) / 3 ; j++) {
            let i0 = indexBuffer[j * 3 + 0];
            let i1 = indexBuffer[j * 3 + 1];
            let i2 = indexBuffer[j * 3 + 2];

            let v0 = positionsBuffer[i0].xyz;
            let v1 = positionsBuffer[i1].xyz;
            let v2 = positionsBuffer[i2].xyz;

            if (rayStepIntersectsTriangle(rayPos, rayDir, v0, v1, v2)) {
                markHit(i0);
                markHit(i1);
                markHit(i2);

                hit = true;
                break;
            }
        }

        if (hit) {
            break;
        }
    }

    // Store ray in the buffer
    let linearIndex = id.x + (id.y * uniforms.raySamples.x);
    if (linearIndex < arrayLength(&rayBuffer)) {
        rayBuffer[linearIndex] = Ray(rayOrigin, stepsTaken, rayDir, stepSize);
    }
}