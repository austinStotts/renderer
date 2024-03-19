struct Parameters {
    radius1: f32,
    sigma1: f32,
    radius2: f32,
    sigma2: f32,
    enable_xdog: u32,
    gfact: f32,
    num_gvf_iterations: u32
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;

@group(1) @binding(0) var<uniform> params: Parameters;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {

    var color = textureSample(inputTexture, sampler0, texcoord);
    return vec4<f32>(color.r, color.g, color.b, 1.0);
}