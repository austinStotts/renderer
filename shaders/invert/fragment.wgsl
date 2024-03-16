
@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var sample: sampler;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(texture, sample, texcoord);
    let red = 1.0 - color.r;
    let green = 1.0 - color.g;
    let blue = 1.0 - color.b;
    return vec4<f32>(red, green, blue, color.a);
}