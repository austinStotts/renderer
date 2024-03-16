@group(0) @binding(0) var srcTexture: texture_2d<f32>;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
    let texelCoord = vec2<i32>(texcoord * vec2<f32>(textureDimensions(srcTexture)));

    let srcPixel: vec4<f32> = textureLoad(srcTexture, texelCoord, 0);

    // Compute the luminance of the pixel
    let luminance = dot(srcPixel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Sobel edge detection
    var gradientX: f32 = 0.0;
    var gradientY: f32 = 0.0;

    // Horizontal Sobel filter
    gradientX += textureLoad(srcTexture, texelCoord + vec2(-1, -1), 0).r * -1.0;
    gradientX += textureLoad(srcTexture, texelCoord + vec2(-1, 0), 0).r * -2.0;
    gradientX += textureLoad(srcTexture, texelCoord + vec2(-1, 1), 0).r * -1.0;
    gradientX += textureLoad(srcTexture, texelCoord + vec2(1, -1), 0).r * 1.0;
    gradientX += textureLoad(srcTexture, texelCoord + vec2(1, 0), 0).r * 2.0;
    gradientX += textureLoad(srcTexture, texelCoord + vec2(1, 1), 0).r * 1.0;

    // Vertical Sobel filter
    gradientY += textureLoad(srcTexture, texelCoord + vec2(-1, -1), 0).r * -1.0;
    gradientY += textureLoad(srcTexture, texelCoord + vec2(0, -1), 0).r * -2.0;
    gradientY += textureLoad(srcTexture, texelCoord + vec2(1, -1), 0).r * -1.0;
    gradientY += textureLoad(srcTexture, texelCoord + vec2(-1, 1), 0).r * 1.0;
    gradientY += textureLoad(srcTexture, texelCoord + vec2(0, 1), 0).r * 2.0;
    gradientY += textureLoad(srcTexture, texelCoord + vec2(1, 1), 0).r * 1.0;

    // Compute the magnitude of the gradient
    let magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);

    // Output the result
    return vec4(magnitude, magnitude, magnitude, 1.0);
}