@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
    const radius1: f32 = 2.0;  // Radius for the first Gaussian
    const sigma1: f32 = radius1 / 3.0;
    const radius2: f32 = 5.0;  // Radius for the second Gaussian
    const sigma2: f32 = radius2 / 3.0;

    let textureSize: vec2<f32> = vec2<f32>(textureDimensions(inputTexture));

    // ----- Gaussian Blur 1 -----
    var blurredImage1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0); 
    var sum1: f32 = 0.0;

    let kernelSize1: i32 = i32(ceil(radius1) * 2.0 + 1.0); 
    for (var offsetX : i32 = -kernelSize1 / 2; offsetX <= kernelSize1 / 2; offsetX++) {
        let samplePos: vec2<f32> = texcoord + vec2<f32>(f32(offsetX) / textureSize.x, 0.0);
        let weight: f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * sigma1 * sigma1)) / (sqrt(2.0 * 3.14159) * sigma1);
        blurredImage1 += textureSample(inputTexture, sampler0, samplePos) * weight;
        sum1 += weight;
    }
    blurredImage1 /= sum1;

    // ----- Gaussian Blur 2 -----
    var blurredImage2: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0); 
    var sum2: f32 = 0.0;

    let kernelSize2: i32 = i32(ceil(radius2) * 2.0 + 1.0); 
    for (var offsetX : i32 = -kernelSize2 / 2; offsetX <= kernelSize2 / 2; offsetX++) {
        let samplePos: vec2<f32> = texcoord + vec2<f32>(f32(offsetX) / textureSize.x, 0.0);
        let weight: f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * sigma2 * sigma2)) / (sqrt(2.0 * 3.14159) * sigma2);
        blurredImage2 += textureSample(inputTexture, sampler0, samplePos) * weight;
        sum2 += weight;
    }
    blurredImage2 /= sum2;

    // ----- Difference of Gaussians -----
    let difference = blurredImage1 - blurredImage2; 

    // ----- Thresholding (optional) -----
    const threshold = 0.05; 
    return vec4<f32>(step(threshold, abs(difference.r)), step(threshold, abs(difference.g)), step(threshold, abs(difference.b)), 1.0);
}










// @group(0) @binding(0) var inputTexture: texture_2d<f32>;
// @group(0) @binding(1) var sampler0: sampler;

// @fragment
// fn main(@builtin(position) FragCoord : vec4<f32>) -> @location(0) vec4<f32> {
//     let textureSize : vec2<f32> = textureDimensions(inputTexture);

//     let radius1 : f32 = 2.0;  // Smaller radius
//     let sigma1 : f32 = radius1 / 3.0; 
//     let radius2 : f32 = 5.0;  // Larger radius
//     let sigma2 : f32 = radius2 / 3.0;

//     // ... (Gaussian blur code from previous example for radius1/sigma1)
//     var blurredImage1 = result; // Store the result of the first blur

//     // ... (Gaussian blur code from previous example for radius2/sigma2)
//     var blurredImage2 = result; // Store the result of the second blur

//     // Difference of Gaussians
//     let difference = blurredImage1 - blurredImage2; 

//     // Thresholding (optional)
//     let threshold = 0.05; 
//     return vec4<f32>(step(threshold, abs(difference.r)), step(threshold, abs(difference.g)), step(threshold, abs(difference.b)), 1.0);
// }











// @group(0) @binding(0) var inputTexture: texture_2d<f32>;
// @group(0) @binding(1) var sampler0: sampler;

// // ... Other parameters (radius1, sigma1, radius2, sigma2)

// @fragment
// fn main(@builtin(position) FragCoord : vec4<f32>) -> @location(0) vec4<f32> {
//     // ... (Remainig shader code)

//     // 1. Gamma Correction
//     let gamma = 2.2;  // Adjust as needed
//     let correctedPixel = pow(textureSample(inputTexture, sampler0, FragCoord.xy / textureSize).rgb, vec3<f32>(1.0 / gamma)); 

//     // 2. Gaussian Blurs (with correctedPixel as input)
//     // ... (Your existing Gaussian blur implementation)

//     // 3. Difference of Gaussians
//     let difference = blurredImage1 - blurredImage2; 

//     // 4. Contrast Equalization (Placeholders - You'll need an implementation)
//     difference = applyHistogramEqualization(difference); 

//     // 5. Noise Suppression (Placeholders - You'll need an implementation)
//     difference = applyBilateralFilter(difference); 

//     // ... (Thresholding, etc.) 
// }



