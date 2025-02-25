struct vsUniforms {
    viewProjection: mat4x4f
};

@group(0) @binding(4) var<uniform> uniforms: vsUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn main(@location(0) position: vec3f, @builtin(vertex_index) vIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    let worldPos = vec4f(position, 1.0);
    output.position = uniforms.viewProjection * worldPos;

    let barycentricCoords = array<vec3f, 3>(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0)
    );

    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(1.0, 1.0, 1.0, 1.0);
}