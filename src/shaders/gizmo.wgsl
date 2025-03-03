@group(0) @binding(0) var<uniform> viewMatrix: mat4x4f;

struct VertexInput {
    @location(0) position: vec4f
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Transform to camera space
    let worldPos = viewMatrix * input.position;
    
    // Scale down for the viewport & move to top-right
    let ndc = worldPos.xy / 8.0; // Shrinks to 1/8 size
    output.position = vec4f(ndc.x + 0.75, ndc.y + 0.75, 0.0, 1.0);
    
    output.color = input.position.rgb + vec3f(0.5, 0.5, 0.5);
    return output;
}

@fragment
fn main_fs(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
