






















use core::result::Result::Ok;
use ::image;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::{
    event::{Event, ElementState, KeyboardInput, VirtualKeyCode, WindowEvent, MouseScrollDelta, MouseButton},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};


use wgpu::util::DeviceExt;
// use wgpu_glyph::{ab_glyph, GlyphBrushBuilder, Section, Text, Layout};










#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 2],
    texcoord: [f32; 2], // Add this line
}


struct PanState {
    is_panning: bool,
    prev_mouse_pos: PhysicalPosition<f64>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
    radius: f32,
    sigma: f32,
}




const ZOOM_SPEED: f32 = 0.1;
// const PAN_SCALAR: f32 = 10000.0;

fn update_vertex_data(zoom_level: &f32, pan_offset: &[f32; 2], window_aspect_ratio: f32, image_aspect_ratio: f32) -> Vec<Vertex> {
    let mut scale_x = *zoom_level;
    let mut scale_y = *zoom_level;

    if window_aspect_ratio > image_aspect_ratio {
        scale_x *= image_aspect_ratio / window_aspect_ratio;
    } else {
        scale_y *= window_aspect_ratio / image_aspect_ratio;
    }

    let vertex_data = [
        Vertex { position: [-scale_x + pan_offset[0], -scale_y + pan_offset[1]], texcoord: [0.0, 1.0] }, // Bottom-left
        Vertex { position: [-scale_x + pan_offset[0], scale_y + pan_offset[1]], texcoord: [0.0, 0.0] },  // Top-left
        Vertex { position: [scale_x + pan_offset[0], scale_y + pan_offset[1]], texcoord: [1.0, 0.0] },   // Top-right
        Vertex { position: [scale_x + pan_offset[0], scale_y + pan_offset[1]], texcoord: [1.0, 0.0] },   // Top-right (repeated)
        Vertex { position: [scale_x + pan_offset[0], -scale_y + pan_offset[1]], texcoord: [1.0, 1.0] },  // Bottom-right
        Vertex { position: [-scale_x + pan_offset[0], -scale_y + pan_offset[1]], texcoord: [0.0, 1.0] }, // Bottom-left (repeated)
    ].to_vec();

    vertex_data
}


fn handle_zoom(delta: &MouseScrollDelta, zoom_level: &mut f32, pan_offset: &[f32; 2], window_aspect_ratio: f32, image_aspect_ratio: f32) -> Vec<Vertex> {
    match delta {
        MouseScrollDelta::LineDelta(_, y) => {
            *zoom_level += y * ZOOM_SPEED;
            *zoom_level = zoom_level.clamp(0.1, 10.0); // Clamp the zoom level between 0.1 and 10.0
        }
        MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => {
            *zoom_level -= (*y as f32) * ZOOM_SPEED;
            *zoom_level = zoom_level.clamp(0.1, 10.0); // Clamp the zoom level between 0.1 and 10.0
        }
    }

    update_vertex_data(zoom_level, pan_offset, window_aspect_ratio, image_aspect_ratio)
}



fn handle_pan(
    curr_mouse_pos: &PhysicalPosition<f64>,
    prev_mouse_pos: &mut PhysicalPosition<f64>,
    zoom_level: &f32,
    pan_offset: &mut [f32; 2],
) {
    let panning_speed_factor = 0.005 / *zoom_level;

    let delta_x = (curr_mouse_pos.x - prev_mouse_pos.x) as f32;
    let delta_y = (curr_mouse_pos.y - prev_mouse_pos.y) as f32;

    pan_offset[0] += delta_x * panning_speed_factor;
    pan_offset[1] += -delta_y * panning_speed_factor;

    *prev_mouse_pos = *curr_mouse_pos;
}





async fn run() {

    println!("RUNNING RENDERER");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    })
    .await
    .unwrap();

    let (device, queue) = adapter
    .request_device(
        &wgpu::DeviceDescriptor {
            label: Some("DEVICE"),
            features: wgpu::Features::default(),
            limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    )
    .await
    .unwrap();
    
    let img = image::load_from_memory(include_bytes!("../images/cat.png")).unwrap();

    let img_ = img.to_rgba8();
    let (mut width, mut height) = img_.dimensions();


    let max_texture_size = device.limits().max_texture_dimension_2d as u32;
    if width > max_texture_size || height > max_texture_size {
        let scale = max_texture_size as f32 / width.max(height) as f32;
        width = (width as f32 * scale) as u32;
        height = (height as f32 * scale) as u32;
    } 

    let image = img.resize(width, height, image::imageops::FilterType::Gaussian).to_rgba8();
    window.set_inner_size(LogicalSize::new(width, height));


    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats[0];
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: Vec::new(),
    };
    surface.configure(&device, &config);

    

    let vertex_data = [
        Vertex { position: [-1.0, -1.0], texcoord: [0.0, 1.0] }, // Bottom-left
        Vertex { position: [-1.0, 1.0], texcoord: [0.0, 0.0] },  // Top-left
        Vertex { position: [1.0, 1.0], texcoord: [1.0, 0.0] },   // Top-right
        Vertex { position: [1.0, 1.0], texcoord: [1.0, 0.0] },   // Top-right (repeated)
        Vertex { position: [1.0, -1.0], texcoord: [1.0, 1.0] },  // Bottom-right
        Vertex { position: [-1.0, -1.0], texcoord: [0.0, 1.0] }, // Bottom-left (repeated)
    ];

    
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: unsafe {
            std::slice::from_raw_parts(
                vertex_data.as_ptr() as *const u8,
                vertex_data.len() * std::mem::size_of::<Vertex>(),
            )
        },
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    



    let image_data = image.into_vec();


    let image_texture = device.create_texture(
        &wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        }
    );
    
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &image_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All
        },
        &image_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * width),
            rows_per_image: std::num::NonZeroU32::new(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1
        },
    );


    let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Texture Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    

    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Texture Bind Group"),
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&image_texture.create_view(&wgpu::TextureViewDescriptor::default())),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                })),
            },
        ],
    });




    let params = Parameters { radius: 3.0, sigma: 1.0 };
    let params_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Parameter buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        }
    );

    let params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("paramerters bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None 
                },
                count: None
            }
        ]
    });

    let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("parameters bind group"),
        layout: &params_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }
        ]
    });




    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("V-Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("../shaders/gaussian-blur/vertex.wgsl"))),
    });

    let frag_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("F-Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("../shaders/gaussian-blur/fragment.wgsl"))),
    });


    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        // bind_group_layouts: &[&texture_bind_group_layout, &palette_bind_group_layout],
        bind_group_layouts: &[&texture_bind_group_layout, &params_bind_group_layout],
        push_constant_ranges: &[],
    });


    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "vert_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &frag_module,
            entry_point: "frag_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });




    let mut zoom_level: f32 = 1.0;
    let mut pan_offset = [0.0, 0.0];
    let mut pan_state = PanState {
        is_panning: false,
        prev_mouse_pos: PhysicalPosition::new(0.0, 0.0),
    };

    let mut current_mouse_position = PhysicalPosition::new(0.0, 0.0);



    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
    
        // println!("START OF EVENT LOOP");

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                let window_size = window.inner_size();
                let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
                let image_aspect_ratio = width as f32 / height as f32;
            
                handle_zoom(&delta, &mut zoom_level, &pan_offset, window_aspect_ratio, image_aspect_ratio);
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { state, button, .. },
                ..
            } => {
                match (state, button) {
                    (ElementState::Pressed, MouseButton::Left) => {
                        pan_state.is_panning = true;
                        pan_state.prev_mouse_pos = current_mouse_position;
                    }
                    (ElementState::Released, MouseButton::Left) => {
                        pan_state.is_panning = false;
                    }
                    _ => {}
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                current_mouse_position = position;
            
                if pan_state.is_panning {
                    handle_pan(&position, &mut pan_state.prev_mouse_pos, &mut zoom_level, &mut pan_offset);
                    // window.request_redraw();
                }
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Space),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                // Reset zoom level and center the image
                zoom_level = 1.0;
                pan_offset = [0.0, 0.0];
            
                let window_size = window.inner_size();
                let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
                let image_aspect_ratio = width as f32 / height as f32;
            
                let new_vertex_data = update_vertex_data(&zoom_level, &pan_offset, window_aspect_ratio, image_aspect_ratio);
                queue.write_buffer(&vertex_buffer, 0, unsafe {
                    std::slice::from_raw_parts(
                        new_vertex_data.as_ptr() as *const u8,
                        vertex_data.len() * std::mem::size_of::<Vertex>(),
                    )
                });
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                let surface_caps = surface.get_capabilities(&adapter);
                let surface_format = surface_caps.formats[0];
    
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: surface_format,
                    width: new_size.width,
                    height: new_size.height,
                    present_mode: surface_caps.present_modes[0],
                    alpha_mode: surface_caps.alpha_modes[0],
                    view_formats: Vec::new(),
                };
    
                surface.configure(&device, &config);
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }


            Event::RedrawRequested(_) => {
                // stats.update();
                match surface.get_current_texture() {
                    Ok(frame) => {
                        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
            
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Render Encoder"),
                        });
            
                        let window_size = window.inner_size();
                        let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
                        let image_aspect_ratio = width as f32 / height as f32;
            
                        let vertex_data = update_vertex_data(&zoom_level, &pan_offset, window_aspect_ratio, image_aspect_ratio);
            
                        queue.write_buffer(&vertex_buffer, 0, unsafe {
                            std::slice::from_raw_parts(
                                vertex_data.as_ptr() as *const u8,
                                vertex_data.len() * std::mem::size_of::<Vertex>(),
                            )
                        });
            
                        {
                            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Render Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 1.0,
                                            g: 1.0,
                                            b: 1.0,
                                            a: 1.0,
                                        }),
                                        store: true,
                                    },
                                })],
                                depth_stencil_attachment: None,
                            });
            
                            render_pass.set_pipeline(&render_pipeline);
                            render_pass.set_bind_group(0, &texture_bind_group, &[]);
                            render_pass.set_bind_group(1, &params_bind_group, &[]);
                            // render_pass.set_bind_group(1, &palette_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                            render_pass.draw(0..6, 0..1);
                        }
            
                        queue.submit(std::iter::once(encoder.finish()));
                        frame.present();
                    }
                    Err(e) => {
                        eprintln!("Error getting current texture: {:?}", e);
                    }
                };
                window.request_redraw();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
    
}



fn main() {
    pollster::block_on(run());
    // show_window()
}
