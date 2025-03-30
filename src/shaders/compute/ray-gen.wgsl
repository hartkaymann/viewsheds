struct compUniforms {
    rayOrigin: vec3f,
    startTheta: f32,
    endTheta: f32,
    startPhi: f32,
    endPhi: f32,
    raySamples: vec2<u32>,
};

struct Ray {
    origin: vec3f,
    length: f32,
    direction: vec3f,
};

@group(0) @binding(0) var<uniform> uniforms: compUniforms;

@group(1) @binding(0) var<storage, read_write> rayBuffer: array<Ray>;

@compute @workgroup_size(__WORKGROUP_SIZE_X__, __WORKGROUP_SIZE_Y__, __WORKGROUP_SIZE_Z__)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.raySamples.x || id.y >= uniforms.raySamples.y) {
        return;
    }

    let index = id.x + id.y * uniforms.raySamples.x;

    let gridSize = vec2f(f32(uniforms.raySamples.x), f32(uniforms.raySamples.y));
    let id2D = vec2f(f32(id.x) / gridSize.x, f32(id.y) / gridSize.y);

    let theta = uniforms.startTheta + id2D.x * (uniforms.endTheta - uniforms.startTheta);
    let phi = acos(mix(cos(uniforms.startPhi), cos(uniforms.endPhi), id2D.y));

    let dir = normalize(vec3f(
        cos(theta) * sin(phi),
        cos(phi),
        sin(theta) * sin(phi)
    ));

    rayBuffer[index] = Ray(uniforms.rayOrigin, 100.0, dir);
}
