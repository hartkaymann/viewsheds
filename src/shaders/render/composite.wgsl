@group(0) @binding(0) var accumSampler: sampler;
@group(0) @binding(1) var accumTexture: texture_2d<f32>;
@group(0) @binding(2) var revealTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn main(@builtin(vertex_index) index: u32) -> VertexOutput {
  var pos = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );

  var uv = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
    vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
  );

  var output: VertexOutput;
  output.position = vec4f(pos[index], 0.0, 1.0);
  output.uv = uv[index];
  return output;
}


@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4f {
  let accum = textureSample(accumTexture, accumSampler, in.uv);
  let reveal = textureSample(revealTexture, accumSampler, in.uv);

  let color = accum.rgb / clamp(accum.a, 1e-4, 5e4);
  let alpha = 1.0 - reveal.r;

  return vec4f(color, alpha);
}