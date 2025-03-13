struct Uniforms {
    modelMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};


@group(0) @binding(2) var<storage, read> visibilityBuffer: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

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
fn main(@location(0) position: vec4f, @location(1) color: vec4f, @builtin(vertex_index) vIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    
    output.position = uniforms.projectionMatrix * uniforms.viewMatrix * position;

    
    let isVisible = getBoolean(vIndex);
    output.color = select(vec4f(1.0, 1.0, 1.0, 1.0), vec4f(1.0, 0.5, 0.3, 1.0), isVisible);

    //let colFac = 0.00392156862;
    //output.color = color * vec4f(colFac, colFac, colFac, 1.0);

    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}