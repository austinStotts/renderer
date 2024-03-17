@group(0) @binding(0) var srcTexture: texture_2d<f32>;

@fragment
fn frag_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
    let texelCoord = vec2<i32>(texcoord * vec2<f32>(textureDimensions(srcTexture)));
    let srcPixel: vec4<f32> = textureLoad(srcTexture, texelCoord, 0);

    // Quantize the pixel to a predefined set of colors
    let palette: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
        vec3<f32>(0.0, 0.0, 0.0), // Black
        vec3<f32>(1.0, 1.0, 1.0), // White
        vec3<f32>(1.0, 0.0, 0.0), // Red
        vec3<f32>(0.0, 1.0, 0.0), // Green
        vec3<f32>(0.0, 0.0, 1.0), // Blue
        vec3<f32>(1.0, 1.0, 0.0), // Yellow
        vec3<f32>(1.0, 0.0, 1.0), // Magenta
        vec3<f32>(0.0, 1.0, 1.0)  // Cyan
    );

    // Find the closest color in the palette
    let black_distance = length(srcPixel.rgb - palette[0]);
    let white_distance = length(srcPixel.rgb - palette[1]);
    let red_distance = length(srcPixel.rgb - palette[2]);
    let green_distance = length(srcPixel.rgb - palette[3]);
    let blue_distance = length(srcPixel.rgb - palette[4]);
    let yellow_distance = length(srcPixel.rgb - palette[5]);
    let magenta_distance = length(srcPixel.rgb - palette[6]);
    let cyan_distance = length(srcPixel.rgb - palette[7]);

    var minDistance: f32 = black_distance;
    var closestColor: vec3<f32> = palette[0];

    if (white_distance < minDistance) {
        minDistance = white_distance;
        closestColor = palette[1];
    }

    if (red_distance < minDistance) {
        minDistance = red_distance;
        closestColor = palette[2];
    }

    if (green_distance < minDistance) {
        minDistance = green_distance;
        closestColor = palette[3];
    }

    if (blue_distance < minDistance) {
        minDistance = blue_distance;
        closestColor = palette[4];
    }

    if (yellow_distance < minDistance) {
        minDistance = yellow_distance;
        closestColor = palette[5];
    }

    if (magenta_distance < minDistance) {
        minDistance = magenta_distance;
        closestColor = palette[6];
    }

    if (cyan_distance < minDistance) {
        minDistance = cyan_distance;
        closestColor = palette[7];
    }

    return vec4<f32>(closestColor, 1.0);
}