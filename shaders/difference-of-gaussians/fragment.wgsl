struct Parameters {
    // radius1: f32,
    sigma1: f32,
    // radius2: f32,
    // sigma2: f32,
    tau: f32,
    gfact: f32,
    epsilon: f32,
    num_gvf_iterations: i32,
    enable_xdog: u32,
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;

@group(1) @binding(0) var<uniform> params: Parameters;
@group(2) @binding(0) var<storage, read> inverse_sqrt_table : array<f32, 256>;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {

    var sigma2 = params.sigma1 / 16.0;
    var radius1 = params.sigma1 * 3.0;
    var radius2 = sigma2 * 2.0;

    var textureSize: vec2<f32> = vec2<f32>(textureDimensions(inputTexture));

    // ----- Gaussian Blur 1 -----
    var blurredImage1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0); 
    var sum1: f32 = 0.0;

    var kernelSize1: i32 = i32(ceil(radius1) * 2.0 + 1.0); 
    for (var offsetX : i32 = -kernelSize1 / 2; offsetX <= kernelSize1 / 2; offsetX++) {
        var samplePos: vec2<f32> = texcoord + vec2<f32>(f32(offsetX) / textureSize.x, 0.0);
        var weight: f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * params.sigma1 * params.sigma1)) / (sqrt(2.0 * 3.14159) * params.sigma1);
        blurredImage1 += textureSample(inputTexture, sampler0, samplePos) * weight;
        sum1 += weight;
    }
    blurredImage1 /= sum1;

    // ----- Gaussian Blur 2 -----
    var blurredImage2: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0); 
    var sum2: f32 = 0.0;

    var kernelSize2: i32 = i32(ceil(radius2) * 2.0 + 1.0); 
    for (var offsetX : i32 = -kernelSize2 / 2; offsetX <= kernelSize2 / 2; offsetX++) {
        var samplePos: vec2<f32> = texcoord + vec2<f32>(f32(offsetX) / textureSize.x, 0.0);
        var weight: f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * sigma2 * sigma2)) / (sqrt(2.0 * 3.14159) * sigma2);
        blurredImage2 += textureSample(inputTexture, sampler0, samplePos) * weight;
        sum2 += weight;
    }
    blurredImage2 /= sum2;

    // ----- Difference of Gaussians -----
    var difference = blurredImage1.r - blurredImage2.r; 

    if (abs(difference) >= params.tau) {
        difference = 1.0 - exp(-difference / params.tau);
    } else {
        difference = 0.0;
    }



    return vec4<f32>(difference, difference, difference, 1.0);

    // ----- Thresholding (optional) -----
    // var threshold = 0.015; 
    // return vec4<f32>(
    //     step(threshold, abs(difference.r)),
    //     step(threshold, abs(difference.r)),
    //     step(threshold, abs(difference.r)),
        // 1.0);
}


    // DEFAULT:
    // var radius1: f32 = 2.0;
    // var sigma1: f32 = radius1 / 3.0;
    // var radius2: f32 = 5.0;
    // var sigma2: f32 = radius2 / 3.0;
    // var blurredImage1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0); 
    // var blurredImage2: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0); 


    // this looked cool:
    // var radius1: f32 = 1.0;
    // var sigma1: f32 = radius1 / 5.0;
    // var radius2: f32 = 3.0;
    // var sigma2: f32 = radius2 / 2.0;
    // var blurredImage1: vec4<f32> = vec4<f32>(0.1, 0.1, 0.1, 0.1); 
    // var blurredImage2: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0); 




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



