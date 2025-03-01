struct vsUniforms {
  viewProjection: mat4x4f
};

@group(0) @binding(2) var<storage, read> visibilityBuffer: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: vsUniforms;

fn getBoolean(index: u32) -> bool {
  let wordIndex = index / 32;
  let bitIndex = index % 32;
  return (visibilityBuffer[wordIndex] & (1u << bitIndex)) != 0;
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>
};

@vertex
fn main(@location(0) position: vec4f, @builtin(vertex_index) vIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    
    output.position = uniforms.viewProjection * position;

    
    let isVisible = getBoolean(vIndex);
    output.color = select(vec4f(1.0, 1.0, 1.0, 1.0), vec4f(1.0, 0.5, 0.3, 1.0), isVisible);

    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}