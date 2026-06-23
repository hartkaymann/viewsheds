struct GridUniforms {
    gVP: mat4x4f,
    cameraWorldPos: vec3f,
};

@group(0) @binding(0)
var<uniform> uniforms: GridUniforms;

// Hardcoded grid configuration. Spacing kept close to the previous grid
// (40-unit minor cells); colors match the previous palette. No X/Z axis lines.
const gridSize = 10000.0;
const gridCellSize = 40.0;
const majorGridDiv = 8.0;       // a major line every N minor cells
const majorLineWidth = 0.05;
const minorLineWidth = 0.02;
const baseColor = vec4f(0.0, 0.0, 0.0, 0.0);
const minorLineColor = vec4f(0.6, 0.6, 0.6, 0.5);
const majorLineColor = vec4f(0.8, 0.8, 0.8, 1.0);

const positions = array<vec3<f32>, 4>(
    vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(-1.0, 0.0, 1.0),
);
const indices = array<u32, 6>(0, 2, 1, 2, 0, 3);

fn composite_over(dst: vec4f, src_color: vec4f, src_t: f32) -> vec4f {
    let a = src_t * src_color.a;
    let out_a = a + dst.a * (1.0 - a);
    if (out_a <= 0.0) { return dst; }
    return vec4f((src_color.rgb * a + dst.rgb * dst.a * (1.0 - a)) / out_a, out_a);
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPos: vec2f,
    @location(1) gridUV: vec2f,
};

@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = positions[indices[vertex_index]] * gridSize;
    pos.x += uniforms.cameraWorldPos.x;
    pos.z += uniforms.cameraWorldPos.z;

    // Snap the UV origin to the camera in whole major-grid steps so the line
    // pattern stays stable (no float wobble) as the camera moves far out.
    let div = max(2.0, round(majorGridDiv));
    let cameraCenteringOffset = floor(uniforms.cameraWorldPos.xz / gridCellSize / div) * div;

    var out: VertexOutput;
    out.worldPos = pos.xz;
    out.gridUV = pos.xz / gridCellSize - cameraCenteringOffset;
    out.position = uniforms.gVP * vec4f(pos, 1.0);
    return out;
}

struct FragOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn main_fs(
    @builtin(position) frag_pos: vec4f,
    @location(0) worldPos: vec2f,
    @location(1) gridUV: vec2f,
) -> FragOut {
    let div = max(2.0, round(majorGridDiv));

    let uvDDX = dpdx(gridUV);
    let uvDDY = dpdy(gridUV);
    let uvDeriv = vec2f(length(vec2f(uvDDX.x, uvDDY.x)), length(vec2f(uvDDX.y, uvDDY.y)));

    // Major lines (every `div` cells)
    let majorUVDeriv = uvDeriv / div;
    let majorLW = majorLineWidth / div;
    let majorDrawWidth = clamp(vec2f(majorLW), majorUVDeriv, vec2f(0.5));
    let majorLineAA = majorUVDeriv * 1.5;
    let majorGridUV = 1.0 - abs(fract(gridUV / div) * 2.0 - 1.0);
    var majorGrid2 = smoothstep(majorDrawWidth + majorLineAA, majorDrawWidth - majorLineAA, majorGridUV);
    majorGrid2 *= clamp(vec2f(majorLW) / majorDrawWidth, vec2f(0.0), vec2f(1.0));
    majorGrid2 = mix(majorGrid2, vec2f(majorLW), clamp(majorUVDeriv * 2.0 - 1.0, vec2f(0.0), vec2f(1.0)));

    // Minor lines (every cell)
    let minorTargetWidth = min(minorLineWidth, majorLineWidth);
    let minorDrawWidth = clamp(vec2f(minorTargetWidth), uvDeriv, vec2f(0.5));
    let minorLineAA = uvDeriv * 1.5;
    let minorGridUV = 1.0 - abs(fract(gridUV) * 2.0 - 1.0);
    var minorGrid2 = smoothstep(minorDrawWidth + minorLineAA, minorDrawWidth - minorLineAA, minorGridUV);
    minorGrid2 *= clamp(vec2f(minorTargetWidth) / minorDrawWidth, vec2f(0.0), vec2f(1.0));
    minorGrid2 = mix(minorGrid2, vec2f(minorTargetWidth), clamp(uvDeriv * 2.0 - 1.0, vec2f(0.0), vec2f(1.0)));

    let minorGrid = mix(minorGrid2.x, 1.0, minorGrid2.y);
    let majorGrid = mix(majorGrid2.x, 1.0, majorGrid2.y);

    var col = baseColor;
    col = composite_over(col, minorLineColor, minorGrid);
    col = composite_over(col, majorLineColor, majorGrid);

    let fade = 1.0 - clamp(distance(worldPos, uniforms.cameraWorldPos.xz) / gridSize, 0.0, 1.0);
    col.a *= pow(fade, 2.5);

    if (col.a <= 0.001) {
        discard;
    }

    var out: FragOut;
    out.color = col;
    // Only write depth where the grid is solid so faint cells don't occlude geometry.
    out.depth = select(1.0, frag_pos.z, col.a >= 0.5);
    return out;
}
