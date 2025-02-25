struct compUniforms {
  cameraPosition: vec4f
};

@group(0) @binding(0) var<storage, read> pointCloud: array<vec3f>;
@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> visibilityBuffer: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> uniforms: compUniforms;

fn rayIntersectsTriangle(rayOrigin: vec3f, rayDir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> bool {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);
    if (abs(a) < 0.00001) { return false; }
    let f = 1.0 / a;
    let s = rayOrigin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = f * dot(edge2, q);
    return t > 0.00001;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&indexBuffer) / 3) { return; }

    let v0 = pointCloud[indexBuffer[index * 3 + 0]];
    let v1 = pointCloud[indexBuffer[index * 3 + 1]];
    let v2 = pointCloud[indexBuffer[index * 3 + 2]];

    let rayOrigin = uniforms.cameraPosition.xyz;
    let rayDir = normalize((v0 + v1 + v2) / 3.0 - rayOrigin);

    let wordIndex = index / 32;
    let bitIndex = index % 32;
    if (rayIntersectsTriangle(rayOrigin, rayDir, v0, v1, v2)) {
        atomicOr(&visibilityBuffer[wordIndex], (1u << bitIndex));
    }

}