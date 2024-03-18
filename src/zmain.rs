use eframe::egui;
use std::sync::{Arc, Mutex};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

#[derive(PartialEq)]
enum AppState {
    GUI,
    Renderer,
}

struct Dialog {
    imagename: String,
    selected_shader_index: usize,
    app_state: Arc<Mutex<AppState>>,
}

impl Default for Dialog {
    fn default() -> Self {
        Self {
            imagename: "cat.png".to_owned(),
            selected_shader_index: 0,
            app_state: Arc::new(Mutex::new(AppState::GUI)),
        }
    }
}

impl eframe::App for Dialog {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("pick an image and shader");
            ui.add_space(15.0);
        
            ui.horizontal(|ui| {
                let name_label = ui.label("select an image: ");
        
                if ui.button("choose File").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                        self.imagename = path.display().to_string();
                    }
                }
        
                ui.label(&self.imagename).labelled_by(name_label.id);
            });
            ui.add_space(15.0);
        
            let shader_options = vec!["invert", "sobel edge detection", "quantization"];
            ui.horizontal(|ui| {
                ui.label("select a shader: ");
                egui::ComboBox::from_label("")
                .selected_text(shader_options[self.selected_shader_index])
                .show_ui(ui, |ui| {
                    
                    ui.selectable_value(&mut self.selected_shader_index, 0, "invert");
                    ui.selectable_value(&mut self.selected_shader_index, 1, "sobel edge detection");
                    ui.selectable_value(&mut self.selected_shader_index, 2, "quantization");
                });
            });
        
            ui.add_space(15.0);

            if ui.button("render").clicked() {
                // Switch to the renderer state
                *self.app_state.lock().unwrap() = AppState::Renderer;  
            }
        });
    }
}

async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Renderer")
        .build(&event_loop)
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window) }.unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
            label: None,
        },
        None,
    ))
    .unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                // Add your wgpu rendering code here
                println!("RUNNING RENDERING CODE");
            }
            _ => (),
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Renderer")
        .build(&event_loop)
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window) }.unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
            label: None,
        },
        None,
    ))
    .unwrap();

    let options = eframe::NativeOptions::default();
    let egui_window = eframe::run_native(
        "renderer",
        options,
        Box::new(|cc| Box::new(Dialog::default())),
    );

    let app_state = Dialog::default().app_state;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                let app_state = app_state.lock().unwrap();
                match *app_state {
                    AppState::Renderer => {
                        // Add your wgpu rendering code here
                        println!("RUNNING RENDERING CODE");
                    }
                    AppState::GUI => egui_window.as_mut().unwrap().update(&event),
                }
            }
            _ => egui_window.as_mut().unwrap().update(&event),
        }
    });
}