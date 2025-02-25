struct vsUniforms {
  viewProjection: mat4x4f
};

struct Ray {
    origin: vec3f,
    direction: vec3f
};

@group(0) @binding(4) var<uniform> uniforms: vsUniforms;
@group(0) @binding(5) var<storage, read> rayBuffer: array<Ray>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>
};

@vertex
fn main_vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    let rayIndex = vertexIndex / 2;
    let isStartPoint = (vertexIndex % 2u) == 0u;

    let ray = rayBuffer[rayIndex];
    let rayPos = select(ray.origin, ray.origin + ray.direction, isStartPoint);
    output.position = uniforms.viewProjection * vec4f(rayPos, 1.0);
    output.color = vec4f(0.0, 1.0, 0.5, 1.0);

    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}