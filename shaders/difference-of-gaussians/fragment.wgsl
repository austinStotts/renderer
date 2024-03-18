@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var sample: sampler;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
    let textureSize : vec2<f32> = vec2<f32>(textureDimensions(texture));
    let radius : f32 = 5.0; // Adjust for desired blur radius
    let sigma : f32 = radius / 3.0; // Standard deviation for Gaussian

    let kernelSize : i32 = i32(ceil(radius) * 2.0 + 1.0); // Odd kernel size

    var result : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum : f32 = 0.0;

    for (var offsetX : i32 = -kernelSize / 2; offsetX <= kernelSize / 2; offsetX++) {
        let samplePos : vec2<f32> = texcoord.xy / textureSize + vec2<f32>(f32(offsetX), 0.0) / textureSize;  // Horizontal pass

        let weight : f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * sigma * sigma)) / (sqrt(2.0 * 3.14159) * sigma);
        result += textureSample(texture, sample, samplePos) * weight;
        sum += weight;
    }

    result /= sum; 

    sum = 0.0;
    for (var offsetY : i32 = -kernelSize / 2; offsetY <= kernelSize / 2; offsetY++) {
        let samplePos : vec2<f32> = texcoord.xy / textureSize + vec2<f32>(f32(offsetY), 0.0) / textureSize;  // Horizontal pass

        let weight : f32 = exp(-(f32(offsetY) * f32(offsetY)) / (2.0 * sigma * sigma)) / (sqrt(2.0 * 3.14159) * sigma);
        result += textureSample(texture, sample, samplePos) * weight;
        sum += weight;
    }

    result /= sum; 

    // Vertical pass (same implementation, swap offsetX and offsetY in the loop)

    return result;
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



