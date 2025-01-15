use egui::{vec2, Vec2};
use rayon::prelude::*;
use egui_file::FileDialog;
use std::{
  ffi::OsStr,
  path::{Path, PathBuf},
};

#[derive(Debug, PartialEq)]
enum Sizes {
    Small,
    Medium,
    Large,
}

pub struct TemplateApp {
    before_texture: egui::TextureHandle,
    after_texture: egui::TextureHandle,
    display_size: Vec2,
    colorshift_amount: u8,
    before_pixels: Vec<u8>,
    after_pixels: Vec<u8>,
    selected_size: Sizes,
    file: Option<PathBuf>,
    open_file_dialog: Option<FileDialog>,
}

impl TemplateApp {  
    /// Called once before the first frame.
    ///       
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {      
        //start with black image
        let initial_size: [usize; 2] = [500,334];
        let len = 4 * initial_size[0] * initial_size[1];
        let mut black_pixels: Vec<u8> = vec![0;len];
        black_pixels.iter_mut().skip(3).step_by(4).for_each(|a|*a = 255); //set alpha
        let black_image = egui::ColorImage::from_rgba_unmultiplied(
            initial_size,
            &black_pixels,
        );

        Self {
            before_texture: cc.egui_ctx.load_texture(
                "before-image",
                black_image.clone(),
                egui::TextureOptions::LINEAR,
            ),
            after_texture: cc.egui_ctx.load_texture(
                "after-image",
                black_image,
                egui::TextureOptions::LINEAR,
            ),
            display_size: vec2(initial_size[0] as f32, initial_size[1] as f32),
            colorshift_amount: 20,
            before_pixels: black_pixels.as_slice().to_vec(),
            after_pixels: black_pixels.as_slice().to_vec(),
            selected_size: Sizes::Medium,
            file: None, //defaults root project dir
            open_file_dialog: None,
        }
    }

    //load the selected file and set pixel vectors and textures
    pub fn load_file(&mut self){
        let image = image::open(self.file.clone().unwrap()).unwrap();
        let image_buffer = image.to_rgba8();
        let pixels = image_buffer.as_flat_samples();
        let size = [image.width() as usize, image.height() as usize];
        //make sure the display size fits on the screen but keep the underlying data
        if image.width()<600 {
            self.display_size = vec2(image.width() as f32, image.height() as f32)
        }else{
            self.display_size = vec2(600.0, image.height() as f32 / image.width() as f32 * 600.0)
        }

        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            size,
            pixels.as_slice(),
        );
    
        self.before_texture.set(color_image.clone(), egui::TextureOptions::LINEAR);
        self.after_texture.set(color_image, egui::TextureOptions::LINEAR);
        self.before_pixels = pixels.as_slice().to_vec();
        self.after_pixels = pixels.as_slice().to_vec();
    }

    pub fn save_file(&mut self){ //saves the base image (need to apply changes from preview)
        image::save_buffer_with_format(self.file.clone().unwrap(), 
                        self.before_pixels.as_slice(), 
                        self.before_texture.size()[0] as u32,
                        self.before_texture.size()[1] as u32,
                image::ExtendedColorType::Rgba8,
            image::ImageFormat::Png)
            .unwrap();
    }

    //set base image as a deep copy of the preview image
    pub fn apply(&mut self){
        self.before_pixels = self.after_pixels.clone();
        self.before_texture.set(
            egui::ColorImage::from_rgba_unmultiplied(self.before_texture.size(), &self.before_pixels),
            egui::TextureOptions::LINEAR
        );
    }
    
    //set preview image as a deep copy of the base image
    pub fn reset(&mut self){
        self.after_pixels = self.before_pixels.clone();
        self.after_texture.set(
            egui::ColorImage::from_rgba_unmultiplied(self.after_texture.size(), &self.after_pixels),
            egui::TextureOptions::LINEAR
        );
    }

    //increase value of a specific color (specified by the offset)
    //uses rayon par_iter
    pub fn colorshift(&mut self, offset: usize){
        self.after_pixels = self.before_pixels.par_iter().enumerate().map(|(i, val)| {
                        //check the offset (right color) and do a checked add
                        if i%4!=offset {*val} else {
                            if (self.colorshift_amount as u32)+(*val as u32)<255 
                                {*val+self.colorshift_amount} 
                            else {255} 
                        }
                    }).collect();
        self.after_texture.set(
            egui::ColorImage::from_rgba_unmultiplied(self.after_texture.size(), &self.after_pixels),
            egui::TextureOptions::LINEAR
        );
    }

    //apply a square kernel in 4 threads
    pub fn convolution_filter(&mut self, kernel: Vec<Vec<i32>>) {
        let size = kernel.len();
        //for normalization:
        let kernel_sum = kernel.clone().into_iter().flatten().reduce(|acc, e| acc + e).unwrap() as f32;
        let kernel_offset = (size/2) as i32; //different between kernel index and rows/cols away
        let width = self.before_texture.size()[0];
        let height = self.before_texture.size()[1];
        //Ensure that division falls evenly on a line break
        let quarterpoint = 4*width*(height/4);
        let midpoint = 4*width*2*(height/4);

        let mut output = vec![0; self.after_pixels.len()]; //where all threads put the new pixel values
        let (half1, half2) = output.split_at_mut(midpoint);
        let (q1, q2 ) = half1.split_at_mut(quarterpoint);
        let (q3, q4 ) = half2.split_at_mut(quarterpoint); //split at 1/4 because slice indexes start at 0 again

        //closure the applies kernel, lots of iteration
        let apply_kernel = |lower, upper, section: &mut[u8]| {
            for i in lower..upper {
                let mut new_val: i32 = 0;
                for r in 0..size{
                    for c in 0..size{
                        //get pixel value from current one and row+col movement
                        let p = pixel_map(self.before_texture.size(), i, r as i32 - kernel_offset, c as i32 - kernel_offset);
                        new_val += self.before_pixels[p] as i32 * kernel[r][c];
                    }
                }
                //normalize, bound, and round new value before setting u8 value
                section[i-lower] = (new_val as f32/kernel_sum + 0.5).min(255.0).max(0.0) as u8;
            }
        };
        //spawn 4 threads
        crossbeam::thread::scope(|scope| {
            scope.spawn(|_| {
                apply_kernel(0, quarterpoint, q1);
            });
            scope.spawn(|_| {
                apply_kernel(quarterpoint, midpoint, q2);
            });
            scope.spawn(|_| {
                apply_kernel(midpoint, midpoint+quarterpoint,q3);
            });
            scope.spawn(|_| {
                apply_kernel(midpoint+quarterpoint, self.before_pixels.len(), q4);
            });
        }).unwrap(); // waits for all threads to finish
        //update image and texture
        self.after_pixels = output;
        self.after_texture.set(
            egui::ColorImage::from_rgba_unmultiplied(self.after_texture.size(), &self.after_pixels),
            egui::TextureOptions::LINEAR
        );
    }

}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {

                //open and save files
                ui.menu_button("File", |ui| {
                    if ui.button("Open file").clicked() {
                        // Show only compatable files
                        let filter = Box::new({
                            let jpg = Some(OsStr::new("jpg"));
                            let jpeg = Some(OsStr::new("jpeg"));
                            let png = Some(OsStr::new("png"));
                            move |path: &Path| -> bool { path.extension() == jpg || path.extension() == jpeg || path.extension() == png }
                        });
                        let mut dialog = FileDialog::open_file(self.file.clone())
                                                    .show_new_folder(false).show_rename(false).show_files_filter(filter);
                        dialog.open();
                        self.open_file_dialog = Some(dialog);
                    }
                    if ui.button("Save file").clicked() {
                        let mut dialog = FileDialog::save_file(self.file.clone())
                                                    .show_new_folder(false).show_rename(false);
                        dialog.open();
                        self.open_file_dialog = Some(dialog);
                    }
                });

                //handle file dialog (every update)
                if let Some(dialog) = &mut self.open_file_dialog {
                    if dialog.show(ctx).selected() {
                        if let Some(file) = dialog.path() {
                            self.file = Some(file.to_path_buf());
                            match dialog.dialog_type() {
                                egui_file::DialogType::SelectFolder => {},
                                egui_file::DialogType::OpenFile => self.load_file(),
                                egui_file::DialogType::SaveFile => self.save_file(),
                            }
                      }
                    }
                }

                ui.separator();
                
                if ui.button("Apply").clicked() {
                    self.apply();
                }
                ui.separator();
                
                if ui.button("Reset preview").clicked() {
                    self.reset();
                }
                ui.separator();
                
                ui.menu_button("Colorshift", |ui| {
                    //various colorshift filters. Adds the offset of the color to the rgba vector
                    if ui.button("red").clicked() {
                        self.colorshift(0);
                    }
                    if ui.button("green").clicked() {
                        self.colorshift(1);
                    }
                    if ui.button("blue").clicked() {
                        self.colorshift(2);
                    }
                });
                ui.add(egui::Slider::new(&mut self.colorshift_amount, 1..=255));
                
                ui.separator();

                ui.menu_button("Convolution filters", |ui| {
                    //various filters with their kernels for each size
                    if ui.button("Gaussian blur").clicked() {
                        let kernel;
                        match self.selected_size {
                            Sizes::Small => {
                                kernel = vec![vec![1,2,1],
                                              vec![2,4,2],
                                              vec![1,2,1]];
                            }
                            Sizes::Medium => {
                                kernel = vec![vec![1,4,7,4,1],
                                              vec![4,16,25,16,4],
                                              vec![7,25,41,51,7],
                                              vec![4,16,25,16,4],
                                              vec![1,4,7,4,1]];
                            },
                            Sizes::Large => {
                                kernel = vec![vec![0,0,1,2,1,0,0],
                                              vec![0,3,13,22,13,3,0],
                                              vec![1,13,59,97,59,13,1],
                                              vec![2,22,97,159,97,22,2],
                                              vec![1,13,59,97,59,13,1],
                                              vec![0,3,13,22,13,3,0],
                                              vec![0,0,1,2,1,0,0]];
                            },
                        }
                        self.convolution_filter(kernel);
                    }

                    if ui.button("Box blur").clicked() {
                        let kernel;
                        match self.selected_size {
                            Sizes::Small => {
                                kernel = vec![vec![1;3];3];
                            }
                            Sizes::Medium => {
                                kernel = vec![vec![1;5];5];
                            },
                            Sizes::Large => {
                                kernel = vec![vec![1;7];7];
                            },
                        }
                        self.convolution_filter(kernel);
                    }

                    if ui.button("Edge enhancment").clicked() {
                        let kernel;
                        match self.selected_size {
                            Sizes::Small => {
                                kernel = vec![vec![-1,-1,-1],
                                              vec![-1,9,-1],
                                              vec![-1,-1,-1]];
                            }
                            Sizes::Medium => {
                                kernel = vec![vec![0,0,-1,0,0],
                                              vec![0,-1,-2,-1,0],
                                              vec![-1,-2,16,-2,-1],
                                              vec![0,-1,-2,-1,0],
                                              vec![0,0,-1,0,0]];
                            },
                            Sizes::Large => {
                                kernel = vec![vec![0,0,-1,-1,-1,0,0],
                                              vec![0,-1,-3,-3,-3,-1,0],
                                              vec![-1,-3,0,7,0,-3,-1],
                                              vec![-1,-3,7,24,7,-3,-1],
                                              vec![-1,-3,0,7,0,-3,-1],
                                              vec![0,-1,-3,-3,-3,-1,0],
                                              vec![0,0,-1,-1,-1,0,0]];
                            },
                        }
                        self.convolution_filter(kernel);
                    }
                });

                //select convolution filter size
                egui::ComboBox::from_label("").selected_text(format!("{:?}", self.selected_size)).
                show_ui(ui, |ui|{
                    ui.selectable_value(&mut self.selected_size, Sizes::Small, "Small");
                    ui.selectable_value(&mut self.selected_size, Sizes::Medium, "Medium");
                    ui.selectable_value(&mut self.selected_size, Sizes::Large, "Large");
                });

                ui.separator();
            });
        });

        egui::SidePanel::left("base_image").show(ctx, |ui| {
            ui.heading("Base image");
            let sized_texture = egui::load::SizedTexture::new( &self.before_texture, self.display_size);
            ui.add(
                egui::Image::new(sized_texture)
            );
        });
        
        egui::SidePanel::right("preview_image").show(ctx, |ui| {
            ui.heading("Preview image");
            let sized_texture = egui::load::SizedTexture::new( &self.after_texture, self.display_size);
            ui.add(
                egui::Image::new(sized_texture)
            );
        });

        egui::CentralPanel::default().show(ctx, |_ui|{});
        
    }
}

//returns pixel value offset by row_offset in the y and col_offset in the x
//handles edges of image (extends)
fn pixel_map(dimensions: [usize;2], index: usize, row_offset: i32, col_offset: i32) -> usize{
    let [width, height] = dimensions;
    let row = (index/4)/width;
    let col = (index/4)%width;
    let pixel_offset = (index%4) as i32;
    let mut new_row = row as i32 + row_offset;
    let mut new_col = col as i32 + col_offset;
    if new_row >= height as i32 {
        new_row = height as i32 -1;
    }else if new_row < 0 {
        new_row = 0;
    }
    
    if new_col >= width as i32 {
        new_col = width as i32 -1;
    }else if new_col < 0 {
        new_col= 0;
    }
    return (new_row*4*width as i32+new_col*4 + pixel_offset) as usize;
}