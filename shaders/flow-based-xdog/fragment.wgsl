struct Parameters {
    radius1: f32,
    sigma1: f32,
    radius2: f32,
    sigma2: f32,
    enable_xdog: u32,
    gfact: f32,
    num_gvf_iterations: i32,
    epsilon: f32
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;

@group(1) @binding(0) var<uniform> params: Parameters;
@group(2) @binding(0) var<storage, read> inverse_sqrt_table : array<f32, 256>;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
   // ... Obtain texture color ...
    var color = textureSample(inputTexture, sampler0, texcoord);
    var original_color = textureSample(inputTexture, sampler0, texcoord);
    if (params.enable_xdog == 1u) {
        // Gradient calculation using Sobel operators
         var sobel_x = array<vec3<f32>, 3>(
             vec3<f32>(-1.0, 0.0, 1.0),
             vec3<f32>(-2.0, 0.0, 2.0),
             vec3<f32>(-1.0, 0.0, 1.0)
         ); 

         var sobel_y = array<vec3<f32>, 3>(
             vec3<f32>(-1.0, -2.0, -1.0),
             vec3<f32>( 0.0,  0.0,  0.0),
             vec3<f32>( 1.0,  2.0,  1.0)
         );

        var dx = 0.0;
        var dy = 0.0;

        for (var i: i32 = -1; i <= 1; i++) {
            for (var j: i32 = -1; j <= 1; j++) {
                var offset : vec2<f32> = vec2<f32>(f32(i), f32(j)) / vec2<f32>(textureDimensions(inputTexture));          
                var sampleColor = textureSample(inputTexture, sampler0, texcoord + offset);
                dx += sampleColor.r * sobel_x[i + 1][j + 1]; 
                dy += sampleColor.r * sobel_y[i + 1][j + 1];
            }
        }

        // Variables to store GVF (initialized to 0.0)
        var u: f32 = 0.0;
        var v: f32 = 0.0;

        // Perform GVF iterations (replace 'num_gvf_iterations' with your desired count)
        for (var i = 0; i < params.num_gvf_iterations; i++) {
            // Update equations (using dx and dy for gradients)
            let u_temp = dx + params.gfact * u;
            let v_temp = dy + params.gfact * v;
            
            // Normalize (sqrt function might require additional implementation)
            let norm_factor = approximate_inversesqrt(u_temp * u_temp + v_temp * v_temp + params.epsilon * params.epsilon);
            u = u_temp * norm_factor;
            v = v_temp * norm_factor;
        }

        // Gaussian Blur Implementation
        var kernelSize1 : i32 = i32(ceil(params.radius1) * 2.0 + 1.0); 
        var kernelSize2 : i32 = i32(ceil(params.radius2) * 2.0 + 1.0);
        var blurredImage1 = color; 
        var blurredImage2 = color; 
        var total_weight: f32 = 0.0;

        // Blur 1 Horizontal Pass
        for (var offsetX : i32 = -kernelSize1 / 2; offsetX <= kernelSize1 / 2; offsetX++) {
            let samplePos: vec2<f32> = texcoord + vec2<f32>(f32(offsetX) / f32(textureDimensions(inputTexture).x), 0.0);
            let weight: f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * params.sigma1 * params.sigma1)) / (sqrt(2.0 * 3.14159) * params.sigma1);
            blurredImage1 += textureSample(inputTexture, sampler0, samplePos) * weight;
            total_weight += weight; // Keep track of total weight
        }


        // Blur 1 Vertical Pass
        for (var offsetY : i32 = -kernelSize1 / 2; offsetY <= kernelSize1 / 2; offsetY++) {
            let samplePos: vec2<f32> = texcoord + vec2<f32>(0.0, f32(offsetY) / f32(textureDimensions(inputTexture).y));
            let weight: f32 = exp(-(f32(offsetY) * f32(offsetY)) / (2.0 * params.sigma1 * params.sigma1)) / (sqrt(2.0 * 3.14159) * params.sigma1);
            blurredImage1 += textureSample(inputTexture, sampler0, samplePos) * weight;
            total_weight += weight; // Keep track of total weight
        }


        // Blur 2 Horizontal Pass
        for (var offsetX : i32 = -kernelSize2 / 2; offsetX <= kernelSize2 / 2; offsetX++) {
            let samplePos: vec2<f32> = texcoord + vec2<f32>(f32(offsetX) / f32(textureDimensions(inputTexture).x), 0.0);
            let weight: f32 = exp(-(f32(offsetX) * f32(offsetX)) / (2.0 * params.sigma2 * params.sigma2)) / (sqrt(2.0 * 3.14159) * params.sigma2);
            blurredImage2 += textureSample(inputTexture, sampler0, samplePos) * weight;
            total_weight += weight; // Keep track of total weight
        }


        // Blur 2 Vertical Pass
        for (var offsetY : i32 = -kernelSize2 / 2; offsetY <= kernelSize2 / 2; offsetY++) {
            let samplePos: vec2<f32> = texcoord + vec2<f32>(0.0, f32(offsetY) / f32(textureDimensions(inputTexture).y));
            let weight: f32 = exp(-(f32(offsetY) * f32(offsetY)) / (2.0 * params.sigma2 * params.sigma2)) / (sqrt(2.0 * 3.14159) * params.sigma2);
            blurredImage2 += textureSample(inputTexture, sampler0, samplePos) * weight;
            total_weight += weight; // Keep track of total weight
        }

        blurredImage1 /= total_weight;
        blurredImage2 /= total_weight;

        var xdog_difference = blurredImage1.r - blurredImage2.r; // Assumes only using red channel


        // Optional Thresholding 
        var threshold = 0.125; 
        if (abs(xdog_difference) < threshold) {
            xdog_difference = 0.0; 
        }

        

        color.r = pow(xdog_difference, 2.0); 
        color.g = color.r;
        color.b = color.r;

        // color.r = abs(dx); // Visualize gradient magnitude X
        // color.g = abs(dy); // Visualize gradient magnitude Y
        // color.b = abs(xdog_difference); // Visualize the XDoG difference
    } // ...
    


    return vec4<f32>(color.r, color.g, color.b, 1.0);
}



fn approximate_inversesqrt(x: f32) -> f32 {
   let normalized_x = clamp(x, 0.0001, 1.0); 
   let index = min(floor(normalized_x * 256.0), 255.0); 
   return inverse_sqrt_table[u32(index)];              
}





