struct compUniforms {
  cameraPosition: vec4f,
  raySamples: vec2<u32>
};

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

@group(0) @binding(0) var<storage, read> pointCloud: array<vec3f>;
@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> visibilityBuffer: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> uniforms: compUniforms;
@group(0) @binding(5) var<storage, read_write> rayBuffer: array<Ray>;

fn rayStepIntersectsTriangle(rayStepPos: vec3<f32>, rayDir: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);

    if (abs(a) < 0.00001) { return false; } // Ray parallel to triangle

    let f = 1.0 / a;
    let s = rayStepPos - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return false; }

    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);
    if (v < 0.0 || (u + v) > 1.0) { return false; }

    let t = f * dot(edge2, q);
    
    return t > 0.00001 && t < 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rayOrigin = vec3f(0.0, 0.0, 0.0); // Example fixed point
    let gridSize = vec2f(f32(uniforms.raySamples.x), f32(uniforms.raySamples.y));
    let id2D = vec2f(f32(id.x), f32(id.y)) / gridSize * 2.0 - vec2f(1.0, 1.0);
    
    let theta = id2D.x * 3.1415926;
    let phi = (id2D.y + 1.0) * 0.5 * 3.1415926;
    let rayDir = normalize(vec3f(
        cos(theta) * sin(phi),  
        cos(phi),
        sin(theta) * sin(phi)   
    ));
    // Store the ray in the buffer
    let linearIndex = id.y * u32(gridSize.x) + id.x;
     rayBuffer[linearIndex] = Ray(rayOrigin, rayDir);
     
    let maxSteps = 50;
    let stepSize = 0.1;
    var rayPos = rayOrigin;

    for (var i = 0; i < maxSteps; i++) {
        rayPos += rayDir * stepSize; // Move the ray forward

        for (var j = 0u; j < arrayLength(&indexBuffer) / 3; j++) {
            let v0 = pointCloud[indexBuffer[j * 3 + 0]];
            let v1 = pointCloud[indexBuffer[j * 3 + 1]];
            let v2 = pointCloud[indexBuffer[j * 3 + 2]];
    
            if (rayStepIntersectsTriangle(rayPos, rayDir, v0, v1, v2)) {
                let wordIndex = j / 32;
        	    let bitIndex = j % 32;
                atomicOr(&visibilityBuffer[wordIndex], (1u << bitIndex));
                break;
            }
        }
    }
}