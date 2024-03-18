struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
};

@vertex
fn vert_main(
    @location(0) position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
) -> VertexOutput {
    // Simple passthrough with texture coordinates
    // var pos = array<vec2<f32>, 4>(
    //     vec2<f32>(-1.0, -1.0),
    //     vec2<f32>(-1.0,  1.0),
    //     vec2<f32>( 1.0, -1.0),
    //     vec2<f32>( 1.0,  1.0));

    // var tex_coords = array<vec2<f32>, 4>(
    //     vec2<f32>(0.0, 0.0),
    //     vec2<f32>(0.0, 1.0),
    //     vec2<f32>(1.0, 0.0),
    //     vec2<f32>(1.0, 1.0));

    // return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.texcoord = texcoord;
    return out;
}

