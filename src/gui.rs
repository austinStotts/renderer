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
        // Send a message to the renderer thread
        println!("RENDER BUTTON");
        if ui.button("render").clicked() { // <-- This check is redundant
            if let Err(e) = self.sender.send(AppState::Renderer) {
                eprintln!("Error sending message to renderer thread: {}", e);
            } else {
                println!("Sent message to renderer thread");
            }
        }
    }

    ui.label(format!("image: [{}], shader: [{}]", self.imagename, shader_options[self.selected_shader_index]));

});