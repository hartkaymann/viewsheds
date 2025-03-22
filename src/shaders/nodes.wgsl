override TREE_DEPTH = 6u;
override BLOCK_SIZE = 128u;

struct Uniforms {
    modelMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};

struct QuadTreeNode {
    position: vec3f,
    childCount: u32,
    size: vec3f,
    startPointIndex: u32,
    pointCount: u32,
    startTriangleIndex: u32,
    triangleCount: u32,
};

const cubeEdges = array<vec3<f32>, 24>(
    // Bottom face edges
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0),

    // Top face edges
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),

    // Vertical edges
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)
);

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> nodeBuffer: array<QuadTreeNode>;
@group(0) @binding(2) var<storage, read> nodeVisibilityBuffer: array<u32>;
@group(0) @binding(3) var<storage, read> rayNodeBuffer: array<u32>;


struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f
};

fn getBoolean(index: u32) -> bool {
  let wordIndex = index / 32;
  let bitIndex = index % 32;
  return (nodeVisibilityBuffer[wordIndex] & (1u << bitIndex)) != 0;
}

@vertex
fn main( 
    @builtin(vertex_index) vertex_index: u32, 
    @builtin(instance_index) instance_index: u32
    ) -> VertexOutput {
    var output: VertexOutput;

    let baseOffset = (1u << (2u * TREE_DEPTH)) / 3u; // bit-shift instead of pow
    let nodeIndex = baseOffset + instance_index;
    let node = nodeBuffer[nodeIndex];

    let localPosition = cubeEdges[vertex_index];
    let worldPosition = (localPosition * node.size) + node.position;
    let position = vec4f(worldPosition.xyz, 1.0);

    output.position = uniforms.projectionMatrix * uniforms.viewMatrix * position;

    let isVisible = getBoolean(instance_index);
    var color = vec4f(1.0, 1.0, 1.0, 0.5); 
    if (getBoolean(instance_index)) {
    color = vec4f(1.0, 0.5, 0.3, 1.0);

    for (var i = 0u; i < 64u; i++) {
        if (rayNodeBuffer[i * BLOCK_SIZE] == nodeIndex) {
            color = vec4f(0.2, 0.5, 1.0, 1.0);
            break;
        }
    }
}
    output.color = color;
    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}