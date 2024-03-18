
struct Parameters {
    radius: f32,
    sigma: f32,
}



@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;

@group(1) @binding(0) var<uniform> params: Parameters;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {

    var kernelSize: i32 = i32(ceil(params.radius) * 2.0 + 1.0);

    var textureSize: vec2<f32> = vec2<f32>(textureDimensions(inputTexture));

    var result: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum: f32 = 0.0;

    // Horizontal pass
    for (var offsetX : i32 = -kernelSize / 2; offsetX <= kernelSize / 2; offsetX++) {
        var samplePos: vec2<f32> = texcoord + vec2<f32>(f32(offsetX) / textureSize.x, 0.0);
        var weight: f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * params.sigma * params.sigma)) / (sqrt(2.0 * 3.14159) * params.sigma);
        result += textureSample(inputTexture, sampler0, samplePos) * weight;
        sum += weight;
    }
    result /= sum;

    // Vertical pass
    sum = 0.0;
    for (var offsetY : i32 = -kernelSize / 2; offsetY <= kernelSize / 2; offsetY++) {
        var samplePos: vec2<f32> = texcoord + vec2<f32>(0.0, f32(offsetY) / textureSize.y);
        var weight: f32 = exp(-(f32(offsetY) * f32(offsetY)) / (2.0 * params.sigma * params.sigma)) / (sqrt(2.0 * 3.14159) * params.sigma);
        result += textureSample(inputTexture, sampler0, samplePos) * weight;
        sum += weight;
    }
    result /= sum;

    return result;
}
