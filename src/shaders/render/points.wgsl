override COLOR: bool = true;

struct Uniforms {
    modelMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};


@group(0) @binding(2) var<storage, read> pointVisibilityBuffer: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> renderMode: u32;  
@group(1) @binding(1) var<storage, read> nodeBuffer: array<vec4<u32>>;
@group(1) @binding(2) var<storage, read> pointToNodeBuffer: array<u32>;

fn getBoolean(index: u32) -> bool {
  let wordIndex = index / 32;
  let bitIndex = index % 32;
  return (pointVisibilityBuffer[wordIndex] & (1u << bitIndex)) != 0;
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

fn randomColor(seed: u32) -> vec3f {
    let x = f32(seed) * 0.123456789;
    let y = f32(seed) * 0.987654321;
    let r = fract(sin(x * 89.42 + y * 4.23) * 43758.5453);
    let g = fract(sin(x * 23.14 + y * 17.73) * 23421.631);
    let b = fract(sin(x * 11.73 + y * 7.97) * 14667.918);
    return vec3f(r, g, b);
}

@vertex
fn main(
  @location(0) position: vec4f, 
  @location(1) color: vec4f, 
  @location(2) classification: u32,
  @builtin(vertex_index) vIndex: u32
  ) -> VertexOutput {
  var output: VertexOutput;
  
  output.position = uniforms.projectionMatrix * uniforms.viewMatrix * position;
 
  switch renderMode {
    case 0u: {  // Vertex color mode
      let colFac = 0.00392156862; // 1/255
      output.color = color * vec4f(colFac, colFac, colFac, 1.0);
    }

    case 1u: {  // Visibility mode
      let isVisible = getBoolean(vIndex);
      output.color = select(vec4f(1.0, 1.0, 1.0, 1.0), vec4f(1.0, 0.5, 0.3, 1.0), isVisible);
    }

    case 2u: { // Quadtree node
      let nodeIndex = pointToNodeBuffer[vIndex];
      output.color = vec4f(randomColor(nodeIndex), 1.0);
    }

    case 3u: { // Classification mode
      let classIndex = classification;
      output.color = vec4f(randomColor(classIndex), 1.0);
    }


    default: {
      output.color = vec4f(1.0, 1.0, 1.0, 1.0);
    }
  }
  
  return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}