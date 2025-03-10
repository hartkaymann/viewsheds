struct Uniforms {
    modelMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};

@group(0) @binding(0) var<storage, read> positionsBuffer: array<vec4f>;
@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    let baseIndex: u32 = instanceIndex * 3;
    let indexOffset = vertexIndex % 3;
    let vertexId = indexBuffer[baseIndex + indexOffset];

    let position = positionsBuffer[vertexId];

    output.position = uniforms.projectionMatrix * uniforms.viewMatrix * position;
    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(1.0, 1.0, 1.0, 1.0);
}