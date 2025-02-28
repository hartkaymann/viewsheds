struct vsUniforms {
    viewProjection: mat4x4f
};

@group(0) @binding(4) var<uniform> uniforms: vsUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn main(@location(0) position: vec4f, @builtin(vertex_index) vIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    output.position = uniforms.viewProjection * position;

    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(1.0, 1.0, 1.0, 1.0);
}