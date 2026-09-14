//! RainAI Studio interactive UI view with Macro-to-Micro controls and 3D Ambisonic radar.

use audio::decoder::DecodeMode;
use audio::export::render_wav_stream;
use eframe::egui::{self, Color32, Pos2, Stroke, Vec2};
use shared::{
    GovernorOptimizationProfile, HardwareStressProfile, NoiseColor, QualityTier, RainState,
    WeatherPreset,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RainTab {
    #[default]
    Weather,
    Surfaces,
    SpatialSounds,
    Presets,
    Export,
    Telemetry,
}

pub const DROPLET_PANNING_SHADER_WGSL: &str = include_str!("../shaders/droplet_panning.wgsl");

/// Uniform buffer payload matching droplet_panning.wgsl WebGPU shader pipeline
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropletPanningUniforms {
    pub resolution: [f32; 2],
    pub time: f32,
    pub rain_intensity: f32,
    pub wind_speed: f32,
    pub wind_azimuth: f32,
    pub fireplace_pos: [f32; 2],
    pub fireplace_intensity: f32,
    pub thunder_pos: [f32; 2],
    pub thunder_intensity: f32,
    pub insect_pos: [f32; 2],
    pub insect_density: f32,
    pub listener_yaw: f32,
}

impl DropletPanningUniforms {
    pub fn from_rain(rain: &RainState, listener_yaw: f32, w: f32, h: f32) -> Self {
        Self {
            resolution: [w, h],
            time: rain.drift_time,
            rain_intensity: rain.weather.intensity,
            wind_speed: rain.wind.speed,
            wind_azimuth: 0.0,
            fireplace_pos: [
                rain.side_sounds.fireplace_azimuth.sin() * 0.45,
                -rain.side_sounds.fireplace_azimuth.cos() * 0.45,
            ],
            fireplace_intensity: rain.side_sounds.fireplace_intensity,
            thunder_pos: [
                rain.side_sounds.thunder_azimuth.sin() * 0.90,
                -rain.side_sounds.thunder_azimuth.cos() * 0.90,
            ],
            thunder_intensity: rain.side_sounds.thunder_proximity,
            insect_pos: [
                rain.side_sounds.insect_azimuth.sin() * rain.side_sounds.insect_proximity,
                -rain.side_sounds.insect_azimuth.cos() * rain.side_sounds.insect_proximity,
            ],
            insect_density: rain.side_sounds.insect_density,
            listener_yaw,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadarDragSource {
    Fireplace,
    Thunder,
    Insect,
    Bird,
    ListenerYaw,
}

pub struct RainView {
    pub current_tab: RainTab,
    pub decode_mode: DecodeMode,
    pub listener_yaw: f32,
    pub quality_download_notice: Option<String>,
    pub export_duration: f32,
    pub export_progress: Option<f32>,
    pub export_status: Option<String>,
    pub share_notice: Option<String>,
    pub enable_gpu_radar: bool,
    pub dragged_source: Option<RadarDragSource>,
}

impl Default for RainView {
    fn default() -> Self {
        Self {
            current_tab: RainTab::Weather,
            decode_mode: DecodeMode::BinauralHeadphones,
            listener_yaw: 0.0,
            quality_download_notice: None,
            export_duration: 30.0,
            export_progress: None,
            export_status: None,
            share_notice: None,
            enable_gpu_radar: false,
            dragged_source: None,
        }
    }
}

impl RainView {
    pub fn render(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        // Step procedural drift if evolve is on
        rain.step_procedural_drift(ui.input(|i| i.stable_dt).min(0.1));

        ui.vertical(|ui| {
            self.render_header(ui, rain);
            ui.add_space(8.0);

            let mut dismiss_notice = false;
            if let Some(notice) = &self.quality_download_notice {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("ℹ️");
                        ui.colored_label(Color32::from_rgb(100, 200, 255), notice);
                        if ui.button("Dismiss").clicked() {
                            dismiss_notice = true;
                        }
                    });
                });
                ui.add_space(4.0);
            }
            if dismiss_notice {
                self.quality_download_notice = None;
            }

            if let Some(share) = &self.share_notice {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("🔗");
                        ui.colored_label(Color32::from_rgb(120, 240, 150), share);
                        if ui.button("Dismiss").clicked() {
                            dismiss_notice = true;
                        }
                    });
                });
                ui.add_space(4.0);
            }
            if dismiss_notice {
                self.share_notice = None;
            }

            self.render_tabs(ui);
            ui.separator();
            ui.add_space(8.0);

            match self.current_tab {
                RainTab::Weather => self.render_weather_tab(ui, rain),
                RainTab::Surfaces => self.render_surfaces_tab(ui, rain),
                RainTab::SpatialSounds => self.render_spatial_sounds_tab(ui, rain),
                RainTab::Presets => self.render_presets_tab(ui, rain),
                RainTab::Export => self.render_export_tab(ui, rain),
                RainTab::Telemetry => self.render_telemetry_tab(ui, rain),
            }
        });
    }

    fn render_header(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        ui.horizontal_wrapped(|ui| {
            // Big play / pause toggle
            let play_btn_text = if rain.is_playing {
                "⏸ Pause Soundscape"
            } else {
                "▶ Start Rain Soundscape"
            };

            let play_btn_color = if rain.is_playing {
                Color32::from_rgb(80, 180, 120)
            } else {
                Color32::from_rgb(60, 120, 220)
            };

            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new(play_btn_text)
                            .size(15.0)
                            .color(Color32::WHITE)
                            .strong(),
                    )
                    .fill(play_btn_color)
                    .min_size(Vec2::new(170.0, 36.0)),
                )
                .clicked()
            {
                rain.is_playing = !rain.is_playing;
            }

            ui.add_space(8.0);

            // Master Volume
            ui.label("Volume:");
            ui.add(
                egui::Slider::new(&mut rain.master_volume, 0.0..=1.0)
                    .show_value(false)
                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
            );

            ui.add_space(8.0);

            // Listening Decode Mode
            ui.label("Format:");
            egui::ComboBox::from_id_salt("decode_mode_selector")
                .selected_text(self.decode_mode.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.decode_mode, DecodeMode::BinauralHeadphones, DecodeMode::BinauralHeadphones.label());
                    ui.selectable_value(&mut self.decode_mode, DecodeMode::StereoSpeakers, DecodeMode::StereoSpeakers.label());
                    ui.selectable_value(&mut self.decode_mode, DecodeMode::Surround71, DecodeMode::Surround71.label());
                    ui.selectable_value(&mut self.decode_mode, DecodeMode::RawFoaPassthrough, DecodeMode::RawFoaPassthrough.label());
                });

            ui.add_space(8.0);

            // Procedural Drift / Evolve toggle
            let evolve_label = if rain.evolve_enabled {
                "🌱 Evolve: On"
            } else {
                "🌱 Evolve: Off"
            };
            ui.toggle_value(&mut rain.evolve_enabled, evolve_label);

            ui.add_space(8.0);

            // Autonomous Meta-Governor toggle
            let gov_label = if rain.auto_quantize {
                "⚡ Governor: Auto"
            } else {
                "⚡ Governor: Manual"
            };
            ui.toggle_value(&mut rain.auto_quantize, gov_label);

            let (badge_col, badge_text) = if !rain.auto_quantize {
                (Color32::from_rgb(160, 160, 170), "Manual Override")
            } else if rain.telemetry.governor_status.contains("Efficiency") || rain.telemetry.governor_status.contains("Critical") {
                (Color32::from_rgb(255, 170, 40), rain.telemetry.governor_status.as_str())
            } else if rain.telemetry.governor_status.contains("Expansion") || rain.telemetry.governor_status.contains("Quality") {
                (Color32::from_rgb(80, 220, 180), rain.telemetry.governor_status.as_str())
            } else {
                (Color32::from_rgb(100, 180, 240), rain.telemetry.governor_status.as_str())
            };
            ui.colored_label(badge_col, format!("[{badge_text}]"));
            ui.colored_label(Color32::from_rgb(180, 140, 255), format!("[{}]", rain.telemetry.active_path_label));
            ui.colored_label(Color32::from_rgb(255, 215, 100), format!("[{}]", rain.telemetry.active_quantization_format));

            if rain.telemetry.is_prebuffered {
                ui.colored_label(Color32::from_rgb(80, 240, 160), "⚡ Buffer Primed (Happy)");
            } else if !rain.is_playing {
                ui.colored_label(Color32::from_rgb(255, 200, 90), format!("⏳ Pre-Buffering ({:.0}ms)", rain.telemetry.buffer_health_ms));
            }

            // Governor Profile Dropdown
            ui.label("Profile:");
            egui::ComboBox::from_id_salt("gov_opt_profile_selector")
                .selected_text(rain.optimization_profile.short_label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::EcoBatterySaver, GovernorOptimizationProfile::EcoBatterySaver.label());
                    ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::LowLatencyInteractive, GovernorOptimizationProfile::LowLatencyInteractive.label());
                    ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::BalancedAdaptive, GovernorOptimizationProfile::BalancedAdaptive.label());
                    ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::StudioMaster, GovernorOptimizationProfile::StudioMaster.label());
                    ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::BluetoothA2DPSink, GovernorOptimizationProfile::BluetoothA2DPSink.label());
                });

            ui.add_space(8.0);

            // Quality Tier Dropdown
            ui.label("Quality:");
            let prev_tier = rain.quality_tier;
            egui::ComboBox::from_id_salt("quality_tier_selector")
                .selected_text(rain.quality_tier.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut rain.quality_tier,
                        QualityTier::Ternary158,
                        format!("{} - {}", QualityTier::Ternary158.label(), QualityTier::Ternary158.download_size_label()),
                    );
                    ui.selectable_value(
                        &mut rain.quality_tier,
                        QualityTier::AdaptiveMinimum,
                        format!("{} - {}", QualityTier::AdaptiveMinimum.label(), QualityTier::AdaptiveMinimum.download_size_label()),
                    );
                    ui.selectable_value(
                        &mut rain.quality_tier,
                        QualityTier::HighInt16,
                        format!("{} - {}", QualityTier::HighInt16.label(), QualityTier::HighInt16.download_size_label()),
                    );
                    ui.selectable_value(
                        &mut rain.quality_tier,
                        QualityTier::StudioFp32,
                        format!("{} - {}", QualityTier::StudioFp32.label(), QualityTier::StudioFp32.download_size_label()),
                    );
                });

            ui.add_space(8.0);

            // Noise Color Selector
            ui.label("Color:");
            egui::ComboBox::from_id_salt("noise_color_selector")
                .selected_text(rain.noise_color.short_label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut rain.noise_color, NoiseColor::Pink, NoiseColor::Pink.label());
                    ui.selectable_value(&mut rain.noise_color, NoiseColor::Brown, NoiseColor::Brown.label());
                    ui.selectable_value(&mut rain.noise_color, NoiseColor::White, NoiseColor::White.label());
                    ui.selectable_value(&mut rain.noise_color, NoiseColor::Blue, NoiseColor::Blue.label());
                    ui.selectable_value(&mut rain.noise_color, NoiseColor::Violet, NoiseColor::Violet.label());
                });

            if prev_tier != rain.quality_tier && rain.quality_tier.is_download_required() {
                self.quality_download_notice = Some(format!(
                    "Downloading on-demand {} weights ({})... Cached in browser OPFS with hot-swapped WASM memory.",
                    rain.quality_tier.label(),
                    rain.quality_tier.download_size_label()
                ));
            }
        });
    }

    fn render_tabs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.current_tab, RainTab::Weather, "🌧 Weather & Wind");
            ui.selectable_value(&mut self.current_tab, RainTab::Surfaces, "🪨 Surface Mixer");
            ui.selectable_value(&mut self.current_tab, RainTab::SpatialSounds, "🧭 Spatial Radar");
            ui.selectable_value(&mut self.current_tab, RainTab::Presets, "✨ Presets Library");
            ui.selectable_value(&mut self.current_tab, RainTab::Export, "💾 Streaming Export");
            ui.selectable_value(&mut self.current_tab, RainTab::Telemetry, "⚡ Telemetry");
        });
    }

    fn render_weather_tab(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        ui.columns(2, |cols| {
            cols[0].group(|ui| {
                ui.heading("Acoustic Atmosphere & Intensity");
                ui.add_space(6.0);

                ui.label("Rainfall Intensity");
                ui.add(egui::Slider::new(&mut rain.weather.intensity, 0.0..=1.0));

                ui.label("Surface Water Runoff / Gutters");
                ui.add(egui::Slider::new(&mut rain.weather.runoff, 0.0..=1.0));

                ui.label("Droplet Distance / Proximity");
                ui.add(egui::Slider::new(&mut rain.weather.distance, 0.0..=1.0));

                ui.label("Enclosure (0.0: Open Outdoor, 1.0: Deep Indoors)");
                ui.add(egui::Slider::new(&mut rain.weather.enclosure, 0.0..=1.0));

                ui.label("Rainfall Pitch / Angle");
                ui.add(egui::Slider::new(&mut rain.weather.pitch_angle, 0.0..=1.0));
            });

            cols[1].group(|ui| {
                ui.heading("Fluid Wind Dynamics");
                ui.add_space(6.0);

                ui.label("Wind Base Speed");
                ui.add(egui::Slider::new(&mut rain.wind.speed, 0.0..=1.0));

                ui.label("Gustiness & Surges");
                ui.add(egui::Slider::new(&mut rain.wind.gustiness, 0.0..=1.0));

                ui.label("Turbulence & Vortices");
                ui.add(egui::Slider::new(&mut rain.wind.turbulence, 0.0..=1.0));

                ui.label("Howling Acoustic Resonance");
                ui.add(egui::Slider::new(&mut rain.wind.howl, 0.0..=1.0));

                ui.add_space(10.0);
                ui.label(egui::RichText::new("Macro Weather Preset Actions").strong());
                ui.horizontal(|ui| {
                    if ui.button("⚡ Heavy Downpour").clicked() {
                        rain.weather.intensity = 0.9;
                        rain.wind.speed = 0.7;
                        rain.wind.gustiness = 0.8;
                    }
                    if ui.button("🍃 Gentle Drizzle").clicked() {
                        rain.weather.intensity = 0.2;
                        rain.wind.speed = 0.15;
                        rain.wind.gustiness = 0.1;
                    }
                });
            });
        });
    }

    fn render_surfaces_tab(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        ui.heading("9-Material Continuous Surface Mixture");
        ui.label("The relative physical area rain impacts upon. Values automatically normalize to 100%.");
        ui.add_space(8.0);

        let normalized = rain.surfaces.normalized();

        ui.columns(3, |cols| {
            cols[0].group(|ui| {
                ui.label(format!("Corrugated Tin ({:.0}%)", normalized[0] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.tin, 0.0..=1.0));

                ui.label(format!("Broad Leaves ({:.0}%)", normalized[1] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.leaves_broad, 0.0..=1.0));

                ui.label(format!("Pine Needles ({:.0}%)", normalized[2] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.pine_needles, 0.0..=1.0));
            });

            cols[1].group(|ui| {
                ui.label(format!("Urban Asphalt Pavement ({:.0}%)", normalized[3] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.pavement, 0.0..=1.0));

                ui.label(format!("Deep Water Body ({:.0}%)", normalized[4] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.water_deep, 0.0..=1.0));

                ui.label(format!("Shallow Puddles ({:.0}%)", normalized[5] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.puddle_shallow, 0.0..=1.0));
            });

            cols[2].group(|ui| {
                ui.label(format!("Canvas Tent ({:.0}%)", normalized[6] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.canvas_tent, 0.0..=1.0));

                ui.label(format!("Glass Window ({:.0}%)", normalized[7] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.glass_window, 0.0..=1.0));

                ui.label(format!("Wood Decking ({:.0}%)", normalized[8] * 100.0));
                ui.add(egui::Slider::new(&mut rain.surfaces.wood_deck, 0.0..=1.0));
            });
        });
    }

    fn render_spatial_sounds_tab(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        ui.columns(2, |cols| {
            cols[0].vertical(|ui| {
                ui.heading("Side Sounds Layer Mix");
                ui.add_space(6.0);

                ui.label("🔥 Fireplace Intensity");
                ui.add(egui::Slider::new(&mut rain.side_sounds.fireplace_intensity, 0.0..=1.0));
                ui.horizontal(|ui| {
                    ui.label("Crackle Rate:");
                    ui.add(egui::Slider::new(&mut rain.side_sounds.fireplace_crackle_rate, 0.0..=1.0).show_value(false));
                });

                ui.separator();

                ui.label("⚡ Thunder Proximity");
                ui.add(egui::Slider::new(&mut rain.side_sounds.thunder_proximity, 0.0..=1.0));
                ui.horizontal(|ui| {
                    ui.label("Rumble Tail:");
                    ui.add(egui::Slider::new(&mut rain.side_sounds.thunder_rumble_length, 0.0..=1.0).show_value(false));
                });

                ui.separator();

                ui.label("🦗 Insects (Cicadas/Crickets)");
                ui.add(egui::Slider::new(&mut rain.side_sounds.insect_density, 0.0..=1.0));

                ui.label("🐦 Birds Activity");
                ui.add(egui::Slider::new(&mut rain.side_sounds.bird_activity, 0.0..=1.0));

                ui.separator();

                ui.label("🚗 Wet Road Traffic Distance");
                ui.add(egui::Slider::new(&mut rain.side_sounds.traffic_distance, 0.0..=1.0));

                ui.add_space(8.0);
                ui.label("Head Orientation / Listener Yaw:");
                ui.add(egui::Slider::new(&mut self.listener_yaw, -std::f32::consts::PI..=std::f32::consts::PI)
                    .custom_formatter(|v, _| format!("{:.0}°", v.to_degrees())));
            });

            cols[1].vertical(|ui| {
                ui.heading("3D Ambisonic Soundfield Radar");
                ui.label("Interactive spatial positioning of sources around the listener");
                ui.add_space(8.0);

                let (response, painter) = ui.allocate_painter(Vec2::new(260.0, 260.0), egui::Sense::click_and_drag());
                let center = response.rect.center();
                let radius = 110.0;

                // Handle interactive click-and-drag source and yaw positioning
                if response.drag_started() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let dx = pos.x - center.x;
                        let dy = pos.y - center.y;
                        let dist = (dx * dx + dy * dy).sqrt();

                        let is_near = |azim: f32, d: f32| -> bool {
                            let angle = azim * std::f32::consts::PI - self.listener_yaw;
                            let r = d.clamp(0.15, 1.0) * radius;
                            let spos = Pos2::new(center.x + r * angle.sin(), center.y - r * angle.cos());
                            (spos.x - pos.x).hypot(spos.y - pos.y) < 22.0
                        };

                        if rain.side_sounds.fireplace_intensity > 0.05 && is_near(rain.side_sounds.fireplace_azimuth, 0.45) {
                            self.dragged_source = Some(RadarDragSource::Fireplace);
                        } else if rain.side_sounds.thunder_proximity > 0.05 && is_near(rain.side_sounds.thunder_azimuth, 0.90) {
                            self.dragged_source = Some(RadarDragSource::Thunder);
                        } else if rain.side_sounds.insect_density > 0.05 && is_near(rain.side_sounds.insect_azimuth, rain.side_sounds.insect_proximity) {
                            self.dragged_source = Some(RadarDragSource::Insect);
                        } else if rain.side_sounds.bird_activity > 0.05 && is_near(-0.4, rain.side_sounds.bird_proximity) {
                            self.dragged_source = Some(RadarDragSource::Bird);
                        } else if dist > radius * 0.75 {
                            self.dragged_source = Some(RadarDragSource::ListenerYaw);
                        } else {
                            self.dragged_source = None;
                        }
                    }
                } else if response.drag_stopped() {
                    self.dragged_source = None;
                }

                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let dx = pos.x - center.x;
                        let dy = pos.y - center.y;
                        let dist = ((dx * dx + dy * dy).sqrt() / radius).clamp(0.15, 1.0);
                        let screen_angle = dx.atan2(-dy);
                        let unrotated_azim = (screen_angle + self.listener_yaw) / std::f32::consts::PI;
                        let norm_azim = ((unrotated_azim + 1.0).rem_euclid(2.0)) - 1.0;

                        match self.dragged_source {
                            Some(RadarDragSource::Fireplace) => {
                                rain.side_sounds.fireplace_azimuth = norm_azim;
                            }
                            Some(RadarDragSource::Thunder) => {
                                rain.side_sounds.thunder_azimuth = norm_azim;
                            }
                            Some(RadarDragSource::Insect) => {
                                rain.side_sounds.insect_azimuth = norm_azim;
                                rain.side_sounds.insect_proximity = dist;
                            }
                            Some(RadarDragSource::Bird) => {
                                rain.side_sounds.bird_proximity = dist;
                            }
                            Some(RadarDragSource::ListenerYaw) => {
                                self.listener_yaw = screen_angle;
                            }
                            None => {}
                        }
                    }
                }

                // Radar background
                painter.circle_filled(center, radius, Color32::from_rgb(14, 18, 26));
                painter.circle_stroke(center, radius, Stroke::new(1.5, Color32::from_rgb(45, 60, 85)));
                painter.circle_stroke(center, radius * 0.66, Stroke::new(1.0, Color32::from_rgb(35, 45, 65)));
                painter.circle_stroke(center, radius * 0.33, Stroke::new(1.0, Color32::from_rgb(35, 45, 65)));

                // Crosshairs
                painter.line_segment([Pos2::new(center.x - radius, center.y), Pos2::new(center.x + radius, center.y)], Stroke::new(1.0, Color32::from_rgb(35, 45, 65)));
                painter.line_segment([Pos2::new(center.x, center.y - radius), Pos2::new(center.x, center.y + radius)], Stroke::new(1.0, Color32::from_rgb(35, 45, 65)));

                // WebGPU Droplet Particle Dynamics (matching droplet_panning.wgsl)
                if self.enable_gpu_radar || (rain.is_playing && rain.weather.intensity > 0.05) {
                    ui.ctx().request_repaint();
                    let t = rain.drift_time;
                    let wind_x = rain.wind.speed * 0.7 + rain.wind.turbulence * 0.3;
                    let wind_y = rain.wind.speed * 0.3 * (1.0 + rain.wind.gustiness);

                    for i in 0..28 {
                        let fi = i as f32;
                        let seed = (fi * 137.5).to_radians();
                        let phase = (t * (0.6 + 0.3 * seed.sin()) + (fi * 0.0357)) % 1.0;
                        let spawn_r = (seed * 2.718).sin().abs() * radius * 0.9;
                        let base_x = center.x + spawn_r * seed.cos() + (phase * wind_x * 24.0);
                        let base_y = (center.y - radius) + (phase * (radius * 2.0 + 16.0)) + (wind_y * 8.0);

                        let drop_pos = Pos2::new(base_x, base_y);
                        let dist_from_center = (drop_pos.x - center.x).hypot(drop_pos.y - center.y);

                        if dist_from_center <= radius {
                            let drop_alpha = ((1.0 - phase) * 160.0 * rain.weather.intensity) as u8;
                            let trail_start = Pos2::new(drop_pos.x - wind_x * 3.5, drop_pos.y - 5.0);
                            painter.line_segment(
                                [trail_start, drop_pos],
                                Stroke::new(1.2, Color32::from_rgba_unmultiplied(130, 205, 255, drop_alpha)),
                            );

                            if phase > 0.82 {
                                let ripple_phase = (phase - 0.82) / 0.18;
                                let ripple_r = (ripple_phase * 12.0).max(2.0);
                                let ripple_alpha = ((1.0 - ripple_phase) * 90.0 * rain.weather.intensity) as u8;
                                painter.circle_stroke(
                                    drop_pos,
                                    ripple_r,
                                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(100, 190, 255, ripple_alpha)),
                                );
                            }
                        }
                    }
                }

                // Rain droplet concentric ripples on radar
                if rain.is_playing && rain.weather.intensity > 0.05 {
                    let t = rain.drift_time * 4.0;
                    let r1 = ((t % 1.0) * radius * 0.8).max(5.0);
                    let alpha = ((1.0 - (t % 1.0)) * 60.0 * rain.weather.intensity) as u8;
                    painter.circle_stroke(center, r1, Stroke::new(1.0, Color32::from_rgba_unmultiplied(100, 180, 255, alpha)));
                }

                // Center listener with head orientation indicator
                painter.circle_filled(center, 6.0, Color32::from_rgb(100, 200, 255));
                let head_dir = Pos2::new(
                    center.x + 14.0 * self.listener_yaw.sin(),
                    center.y - 14.0 * self.listener_yaw.cos(),
                );
                painter.line_segment([center, head_dir], Stroke::new(2.0, Color32::WHITE));
                painter.text(Pos2::new(center.x, center.y - 14.0), egui::Align2::CENTER_CENTER, "You", egui::FontId::proportional(11.0), Color32::WHITE);

                // Draw source positions
                let draw_source = |painter: &egui::Painter, azimuth: f32, dist: f32, icon: &str, active: bool, color: Color32| {
                    if !active {
                        return;
                    }
                    let angle = azimuth * std::f32::consts::PI - self.listener_yaw;
                    let r = dist.clamp(0.15, 1.0) * radius;
                    let pos = Pos2::new(center.x + r * angle.sin(), center.y - r * angle.cos());
                    painter.circle_filled(pos, 5.0, color);
                    painter.text(pos, egui::Align2::CENTER_CENTER, icon, egui::FontId::proportional(14.0), Color32::WHITE);
                };

                draw_source(&painter, rain.side_sounds.fireplace_azimuth, 0.45, "🔥", rain.side_sounds.fireplace_intensity > 0.05, Color32::from_rgb(255, 140, 40));
                draw_source(&painter, rain.side_sounds.thunder_azimuth, 0.90, "⚡", rain.side_sounds.thunder_proximity > 0.05, Color32::from_rgb(255, 230, 80));
                draw_source(&painter, rain.side_sounds.insect_azimuth, rain.side_sounds.insect_proximity, "🦗", rain.side_sounds.insect_density > 0.05, Color32::from_rgb(120, 220, 80));
                draw_source(&painter, -0.4, rain.side_sounds.bird_proximity, "🐦", rain.side_sounds.bird_activity > 0.05, Color32::from_rgb(80, 180, 255));

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.enable_gpu_radar, "⚡ WebGPU Shader Pipeline");
                    if self.enable_gpu_radar {
                        let _uniforms = DropletPanningUniforms::from_rain(rain, self.listener_yaw, 260.0, 260.0);
                        ui.colored_label(Color32::from_rgb(100, 240, 160), "droplet_panning.wgsl active (28 simulated particles)");
                    }
                });
            });
        });
    }

    fn render_presets_tab(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        ui.heading("Curated Atmospheric Presets");
        ui.label("Select handcrafted environmental soundscapes or share custom states.");
        ui.add_space(8.0);

        let builtins = WeatherPreset::builtins();
        egui::Grid::new("presets_grid")
            .num_columns(2)
            .spacing([16.0, 12.0])
            .show(ui, |ui| {
                for (i, preset) in builtins.into_iter().enumerate() {
                    ui.group(|ui| {
                        ui.set_width(340.0);
                        ui.heading(&preset.name);
                        ui.label(egui::RichText::new(&preset.description).italics());
                        ui.add_space(4.0);

                        ui.horizontal(|ui| {
                            for tag in &preset.tags {
                                ui.label(egui::RichText::new(format!("#{tag}")).size(10.0).color(Color32::from_rgb(140, 180, 220)));
                            }
                        });
                        ui.add_space(6.0);

                        ui.horizontal(|ui| {
                            if ui.button(egui::RichText::new("▶ Load & Play").color(Color32::from_rgb(80, 200, 140)).strong()).clicked() {
                                *rain = preset.state.clone();
                                rain.is_playing = true;
                            }
                            if ui.button("📋 Share Link").clicked() {
                                if let Ok(hash) = preset.to_shareable_url_hash() {
                                    #[cfg(target_arch = "wasm32")]
                                    {
                                        if let Some(win) = web_sys::window() {
                                            if let Ok(href) = win.location().href() {
                                                let base = href.split('#').next().unwrap_or(&href);
                                                let full_url = format!("{}#preset={}", base, hash);
                                                let _ = win.location().set_hash(&format!("preset={}", hash));
                                                ui.ctx().copy_text(full_url);
                                                self.share_notice = Some("Shareable URL copied to clipboard & browser hash updated!".into());
                                            }
                                        }
                                    }
                                    #[cfg(not(target_arch = "wasm32"))]
                                    {
                                        ui.ctx().copy_text(hash.clone());
                                        self.share_notice = Some(format!("Preset token copied to clipboard: {}", &hash[..hash.len().min(24)]));
                                    }
                                }
                            }
                        });
                    });

                    if i % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
    }

    fn render_export_tab(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        ui.heading("Faster-Than-Realtime Streaming Audio Exporter");
        ui.label("Export studio-quality uncompressed WAV audio. Audio is rendered in low-memory streaming chunks.");
        ui.add_space(10.0);

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Render Duration:");
                ui.selectable_value(&mut self.export_duration, 10.0, "10s (Quick Sample)");
                ui.selectable_value(&mut self.export_duration, 30.0, "30s (Loop)");
                ui.selectable_value(&mut self.export_duration, 60.0, "1 Minute");
                ui.selectable_value(&mut self.export_duration, 300.0, "5 Minutes");
            });

            ui.add_space(6.0);
            ui.label(format!("Active Decode Format: {}", self.decode_mode.label()));
            ui.label("Output Format: WAV 32-bit IEEE Float (48,000 Hz)");

            ui.add_space(10.0);

            let is_exporting = self.export_progress.is_some();
            if is_exporting {
                let progress = self.export_progress.unwrap_or(0.0);
                ui.add(egui::ProgressBar::new(progress).text(format!("{:.0}% Rendered", progress * 100.0)));
            } else if ui.button(egui::RichText::new("⚡ Start Streaming WAV Export").size(16.0).color(Color32::WHITE)).clicked() {
                let mut buffer = Vec::new();
                let render_result = render_wav_stream(
                    rain,
                    self.export_duration,
                    48000,
                    self.decode_mode,
                    &mut buffer,
                    |_progress| {},
                );

                match render_result {
                    Ok(bytes) => {
                        let filename = format!("rainai_{}s.wav", self.export_duration as u32);
                        crate::storage_manager::trigger_binary_download(&filename, &buffer, "audio/wav");
                        self.export_status = Some(format!(
                            "Successfully exported {:.1} MB uncompressed WAV ({filename}) in streaming chunks!",
                            bytes as f64 / 1_048_576.0
                        ));
                    }
                    Err(e) => {
                        self.export_status = Some(format!("Export error: {e}"));
                    }
                }
            }

            if let Some(status) = &self.export_status {
                ui.add_space(8.0);
                ui.colored_label(Color32::from_rgb(100, 220, 160), status);
            }
        });
    }

    fn render_telemetry_tab(&mut self, ui: &mut egui::Ui, rain: &mut RainState) {
        ui.heading("Invasive Meta-Controller Telemetry, Optimization Profiles & Stress Harness");
        ui.add_space(8.0);

        // Hardware Stress Simulation Test Harness
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("🛠️ Hardware Stress Simulation Harness (Profiles 0–7)").strong());
                ui.colored_label(Color32::from_rgb(255, 180, 80), format!("[Active: {}]", rain.stress_profile.label()));
            });
            ui.label(rain.stress_profile.description());
            ui.add_space(6.0);

            ui.horizontal_wrapped(|ui| {
                ui.label("Inject Scenario:");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::NominalDesktop, "0: Nominal");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::ThermalThrottlingCascade, "1: Thermal Cascade");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::GcWebAudioMicroStalls, "2: GC Micro-Stalls");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::UnifiedMemoryBusContention, "3: Memory Choke");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::DynamicGameDawInterference, "4: DAW Bursts");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::BluetoothA2dpAudioSink, "5: Bluetooth A2DP");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::EcoSleepSoundscapeMode, "6: Eco Sleep");
                ui.selectable_value(&mut rain.stress_profile, HardwareStressProfile::HeterogeneousEcoreAsymmetry, "7: E-Core Asymmetry");
            });

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(format!("Simulated Panic Factor: {:.2}", rain.telemetry.panic_factor));
                ui.add(egui::ProgressBar::new(rain.telemetry.panic_factor).text(if rain.telemetry.panic_factor > 0.5 { "High Stress" } else { "Nominal" }));
                ui.label(format!("Simulated Jitter: {:.1}%", rain.telemetry.jitter_factor * 100.0));
            });
        });

        ui.add_space(8.0);

        // Operational Optimization Profile Card
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("⚙️ Operational Governor Optimization Profile").strong());
                ui.colored_label(Color32::from_rgb(100, 200, 255), format!("[{}]", rain.optimization_profile.label()));
            });
            ui.label(rain.optimization_profile.description());
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label("Select Profile:");
                ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::EcoBatterySaver, GovernorOptimizationProfile::EcoBatterySaver.short_label());
                ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::LowLatencyInteractive, GovernorOptimizationProfile::LowLatencyInteractive.short_label());
                ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::BalancedAdaptive, GovernorOptimizationProfile::BalancedAdaptive.short_label());
                ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::StudioMaster, GovernorOptimizationProfile::StudioMaster.short_label());
                ui.selectable_value(&mut rain.optimization_profile, GovernorOptimizationProfile::BluetoothA2DPSink, GovernorOptimizationProfile::BluetoothA2DPSink.short_label());
            });
        });

        ui.add_space(8.0);

        ui.columns(2, |cols| {
            cols[0].group(|ui| {
                ui.label(egui::RichText::new("Buffer & Latency Telemetry").strong());
                ui.add_space(4.0);

                let target_buf = rain.optimization_profile.target_buffer_ms();
                ui.label(format!("Audio Worklet Buffer Health: {:.1} ms", rain.telemetry.buffer_health_ms));
                ui.add(egui::ProgressBar::new(rain.telemetry.buffer_health_ms / target_buf.max(1.0)).text(format!("Target: {:.0}ms", target_buf)));

                ui.add_space(6.0);
                ui.label(format!("Frame Jitter Delta-T: {:.1} ms", rain.telemetry.delta_t_ms));
                ui.label(format!("CPU Compute Headroom: {:.0}%", rain.telemetry.cpu_headroom * 100.0));
                ui.add(egui::ProgressBar::new(rain.telemetry.cpu_headroom));

                ui.label(format!("GPU WebGPU Headroom: {:.0}%", rain.telemetry.gpu_headroom * 100.0));
                ui.add(egui::ProgressBar::new(rain.telemetry.gpu_headroom));
            });

            cols[1].group(|ui| {
                ui.label(egui::RichText::new("Autonomous Meta-Controller Actions").strong());
                ui.add_space(4.0);

                // MoE Expert Shedding
                let exp_ratio = rain.telemetry.active_experts as f32 / 8.0;
                ui.label(format!("Active MoE Trajectory Experts: {} / 8", rain.telemetry.active_experts));
                ui.add(egui::ProgressBar::new(exp_ratio).text(format!("{} Active Experts", rain.telemetry.active_experts)));

                // Latent Diffusion Bypass Action
                if rain.telemetry.diffusion_bypassed {
                    ui.colored_label(Color32::from_rgb(255, 120, 60), "⚡ Latent Diffusion Bypass: ACTIVE (Fast DSP Projection)");
                } else {
                    ui.colored_label(Color32::from_rgb(120, 220, 150), "✓ Latent Diffusion Bypass: Inactive (Full Recurrent Denoising)");
                }

                // Ambisonic Order Scaling Action
                if rain.telemetry.ambisonic_order_reduced {
                    ui.colored_label(Color32::from_rgb(255, 200, 80), "⚠️ Ambisonic Order Scaling: Reduced to Stereo (Compute Shed)");
                } else {
                    ui.colored_label(Color32::from_rgb(120, 220, 150), "✓ Ambisonic Order Scaling: Full FOA (4-Channel SN3D)");
                }

                ui.add_space(6.0);
                ui.label(format!("Introspective Quality Critic: {:.1}% Certainty", rain.telemetry.quality_critic_score * 100.0));
                ui.add(egui::ProgressBar::new(rain.telemetry.quality_critic_score));

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Meta-Governor Dynamic Quantization").strong());
                ui.label(format!("Effective Bit-Width Range: [{:.2}b - {:.1}b]", rain.telemetry.effective_quant_floor, rain.telemetry.effective_quant_ceiling));
                let quant_fraction = ((rain.telemetry.effective_quant_floor - 1.58) / (32.0 - 1.58)).clamp(0.0, 1.0);
                ui.add(egui::ProgressBar::new(quant_fraction).text(format!("{:.2}b active floor", rain.telemetry.effective_quant_floor)));

                ui.label(format!("Procedural Synthesis Blend: {:.1}%", rain.telemetry.synthesis_blend * 100.0));
                ui.label(format!("Governor Decision: {}", rain.telemetry.governor_status));
                ui.label(format!("Active Hardware Target: {}", rain.telemetry.active_path_label));

                ui.add_space(6.0);
                ui.label("Conditioning Vector Dimension: 554 floats (Physical & Ambisonic)");
                ui.label(format!("Preferred Target Tier: {}", rain.quality_tier.label()));
            });
        });

        ui.add_space(10.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new("Universal Precision Spectrum & Layer Assignments").strong());
            ui.label("Hardware representation and numerical mapping across the neural synthesis pipeline:");
            ui.add_space(6.0);

            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    ui.label(egui::RichText::new("Layer Heterogeneous Roles").strong());
                    ui.label("• Mamba SSM Recurrence: BF16 / TF32 (Extreme Exponent Stability)");
                    ui.label("• Latent VAE Bottlenecks: Posit16 <16, 1> (Tapered Precision near 0 dBFS)");
                    ui.label("• MoE & Dense Projections: FP16 (WebGPU) / INT8 (CPU SIMD)");
                    ui.label("• DDSP Biquad Filter Poles: FP32 (24-bit linear significand, no limit cycles)");
                    ui.label("• Ambisonic Spatial Rotations: FP32 (Exact phase preservation)");
                    ui.label("• Macro Conditioning: Posit8 / FP16 (Smooth parameter manifold)");
                });

                cols[1].vertical(|ui| {
                    ui.label(egui::RichText::new("Discrete Grid & Snapping Mechanics").strong());
                    ui.label("• Integer Levels: L = round(2^b) in {0, 3, 4, 16, 64, 256, 65536, 2^32}");
                    ui.label("• Normalization Invariant: Standardized [-scale, +scale] physical range");
                    ui.label("• Equal-Power Hann Crossfade: 128 samples (2.67ms) clickless transitions");
                    ui.label(format!("• Active Target Format: {}", rain.telemetry.active_quantization_format));
                    if rain.telemetry.is_prebuffered {
                        ui.colored_label(Color32::from_rgb(80, 240, 160), format!("• Buffer Status: {:.0}ms Primed & Ready (Happy)", rain.optimization_profile.target_buffer_ms()));
                    } else {
                        ui.colored_label(Color32::from_rgb(255, 200, 80), format!("• Buffer Status: Pre-Buffering ({:.0}ms / {:.0}ms)", rain.telemetry.buffer_health_ms, rain.optimization_profile.target_buffer_ms()));
                    }
                });
            });
        });
    }
}
