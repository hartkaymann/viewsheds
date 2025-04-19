
struct GridUniforms {
    gVP: mat4x4f,
    cameraWorldPos: vec3f,
};

const positions = array<vec3<f32>, 4>(
    // Bottom face edges
    vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(-1.0, 0.0, 1.0),
);
const indices = array<u32, 6>(0, 2, 1, 2, 0, 3);

const gridSize = 100.0;
const gridMinPixelsBetweenCells = 2.0;
const gridCellSize = 0.025;
const gridColorThin = vec4(0.5, 0.5, 0.5, 1.0);
const gridColorThick = vec4(0.0, 0.0, 0.0, 1.0);

@group(0) @binding(0)
var<uniform> uniforms: GridUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPos: vec3f,
};

fn log10(x: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    return log(x) / log(10.0);
}

fn satf(x: f32) -> f32 {
    return max(0.0, min(1.0, x));
}

fn satv(x: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(satf(x.x), satf(x.y));
}

fn max2(v: vec2<f32>) -> f32 {
    return max(v.x, v.y);
}

@vertex
fn main( 
    @builtin(vertex_index) vertex_index: u32 
) -> VertexOutput {
    var pos = positions[indices[vertex_index]] * gridSize;

    pos.x += uniforms.cameraWorldPos.x;
    pos.z += uniforms.cameraWorldPos.z;

    let worldPos = pos;
    let pos4 = vec4f(pos, 1.0);

    var out: VertexOutput;
    out.position = uniforms.gVP * pos4;
    out.worldPos = worldPos;

    return out;
}

@fragment
fn main_fs(
    @location(0) worldPos: vec3f,
) -> @location(0) vec4f {
    let dvx = vec2f(dpdx(worldPos.x), dpdy(worldPos.x));
    let dvy = vec2f(dpdx(worldPos.z), dpdy(worldPos.z));

    let lx = length(dvx);
    let ly = length(dvy);

    var dudv = vec2f(lx, ly);

    let l = length(dudv);

    let lod = max(0.0, log10(l * gridMinPixelsBetweenCells / gridCellSize) + 1.0);

    let gridCellSizeLod0 = gridCellSize * pow(10.0, floor(lod));
    let gridCellSizeLod1 = gridCellSizeLod0 * 10.0;
    let gridCellSizeLod2 = gridCellSizeLod1 * 10.0;

    dudv *= 4.0;

    var mod_div_dudv = (worldPos.xz % gridCellSizeLod0) / dudv;
    let lod0a = max2(vec2f(1.0) - abs(satv(mod_div_dudv) * 2.0 - vec2f(1.0)) );

    mod_div_dudv = (worldPos.xz % gridCellSizeLod1) / dudv;
    let lod1a = max2(vec2f(1.0) - abs(satv(mod_div_dudv) * 2.0 - vec2f(1.0)) );

    mod_div_dudv = (worldPos.xz % gridCellSizeLod2) / dudv;
    let lod2a = max2(vec2f(1.0) - abs(satv(mod_div_dudv) * 2.0 - vec2f(1.0)) );
    
    let LOD_fade = fract(lod);
    var color: vec4f;

    if (lod2a > 0.0) {
        color = gridColorThick;
        color.a *= lod2a;
    } else {
        if (lod1a > 0.0) {
            color = mix(gridColorThick, gridColorThin, LOD_fade);
	        color.a *= lod1a;
        } else {
            color = gridColorThin;
	        color.a *= (lod0a * (1.0 - LOD_fade));
        }
    }

    let opacityFalloff = (1.0 - satf(length(worldPos.xz - uniforms.cameraWorldPos.xz) / gridSize));

    color.a *= opacityFalloff;

    // return color;
    return vec4f(1.0, 1.0, 1.0, 1.0);
}