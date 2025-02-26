struct vsUniforms {
  viewProjection: mat4x4f
};

struct Ray {
    origin: vec3f,
    _padding1: f32,
    direction: vec3f,
    _padding2: f32
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

    let rayIndex = vertexIndex >> 1; // Fast division by 2
    let isStartPoint = (vertexIndex & 1u) == 0u; // Bitwise check for even/odd

    if (rayIndex < arrayLength(&rayBuffer)) {
        let ray = rayBuffer[rayIndex];
        let rayPos = select(ray.origin, ray.origin + ray.direction, isStartPoint);
        output.position = uniforms.viewProjection * vec4f(rayPos, 1.0);
        output.color = vec4f(0.0, 1.0, 0.5, 1.0);
    } else {
        // Handle out-of-bounds case (optional)
        output.position = vec4f(0.0, 0.0, 0.0, 1.0);
        output.color = vec4f(1.0, 0.0, 0.0, 1.0); // Red color for debugging
    }

    return output;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}