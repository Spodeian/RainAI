//! Item list, progress metrics, and item input components.

use crate::{ScreenConstraints, TemplateApp};
use eframe::egui;
use shared::Priority;

pub fn render_summary_cards(
    app: &mut TemplateApp,
    ui: &mut egui::Ui,
    constraints: &ScreenConstraints,
) {
    let total = app.state.collection.total_count();
    let completed = app.state.collection.completed_count();
    let ratio = app.state.collection.completion_ratio();

    egui::Frame::group(ui.style())
        .inner_margin(if constraints.is_mobile { 10.0 } else { 16.0 })
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Progress Overview").strong().size(16.0));
                    ui.label(format!(
                        "Completed {} of {} items ({:.0}%)",
                        completed,
                        total,
                        ratio * 100.0
                    ));
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let bar_width = (ui.available_width() - 8.0).clamp(100.0, 300.0);
                    let pb = ui.add(
                        egui::ProgressBar::new(ratio)
                            .text(format!("{:.0}%", ratio * 100.0))
                            .desired_width(bar_width),
                    );
                    pb.widget_info(move || {
                        egui::WidgetInfo::labeled(
                            egui::WidgetType::ProgressIndicator,
                            true,
                            format!(
                                "Task completion progress: {:.0}% ({} of {} completed)",
                                ratio * 100.0,
                                completed,
                                total
                            ),
                        )
                    });
                });
            });
        });
}

pub fn render_new_item_form(
    app: &mut TemplateApp,
    ui: &mut egui::Ui,
    constraints: &ScreenConstraints,
) {
    egui::Frame::group(ui.style())
        .inner_margin(if constraints.is_mobile { 10.0 } else { 16.0 })
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Add New Item").strong().size(15.0));
            ui.add_space(6.0);

            let mut submit_new_item = false;

            if constraints.is_mobile {
                ui.vertical(|ui| {
                    let title_resp = ui.add(
                        egui::TextEdit::singleline(&mut app.new_item_title)
                            .hint_text("Item title...")
                            .desired_width(f32::INFINITY),
                    );
                    if title_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        submit_new_item = true;
                    }
                    ui.add_space(4.0);
                    let desc_resp = ui.add(
                        egui::TextEdit::singleline(&mut app.new_item_description)
                            .hint_text("Item description (optional)...")
                            .desired_width(f32::INFINITY),
                    );
                    if desc_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        submit_new_item = true;
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("mobile_priority_dropdown")
                            .selected_text(format!("Priority: {}", app.new_item_priority.label()))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut app.new_item_priority,
                                    Priority::Low,
                                    "Low",
                                );
                                ui.selectable_value(
                                    &mut app.new_item_priority,
                                    Priority::Medium,
                                    "Medium",
                                );
                                ui.selectable_value(
                                    &mut app.new_item_priority,
                                    Priority::High,
                                    "High",
                                );
                            });

                        if ui.button("➕ Add Item").clicked() {
                            submit_new_item = true;
                        }
                    });
                });
            } else {
                ui.horizontal(|ui| {
                    let title_resp = ui.add(
                        egui::TextEdit::singleline(&mut app.new_item_title)
                            .hint_text("Item title...")
                            .desired_width(220.0),
                    );
                    if title_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        submit_new_item = true;
                    }
                    let desc_resp = ui.add(
                        egui::TextEdit::singleline(&mut app.new_item_description)
                            .hint_text("Description (optional)...")
                            .desired_width(320.0),
                    );
                    if desc_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        submit_new_item = true;
                    }
                    egui::ComboBox::from_id_salt("desktop_priority_dropdown")
                        .selected_text(app.new_item_priority.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut app.new_item_priority, Priority::Low, "Low");
                            ui.selectable_value(
                                &mut app.new_item_priority,
                                Priority::Medium,
                                "Medium",
                            );
                            ui.selectable_value(&mut app.new_item_priority, Priority::High, "High");
                        });

                    if ui.button("➕ Add Item").clicked() {
                        submit_new_item = true;
                    }
                });
            }

            if submit_new_item && !app.new_item_title.trim().is_empty() {
                app.state.collection.add(
                    app.new_item_title.trim(),
                    app.new_item_description.trim(),
                    app.new_item_priority,
                );
                app.new_item_title.clear();
                app.new_item_description.clear();
                app.persist_state();
            }
        });
}

pub fn render_item_list(app: &mut TemplateApp, ui: &mut egui::Ui, constraints: &ScreenConstraints) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Items").heading());

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("Clear Completed")
                .on_hover_text("Remove all finished items")
                .clicked()
            {
                app.state.collection.clear_completed();
                app.persist_state();
            }

            egui::ComboBox::from_id_salt("filter_priority_dropdown")
                .selected_text(match app.filter_priority {
                    None => "All Priorities",
                    Some(p) => p.label(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut app.filter_priority, None, "All Priorities");
                    ui.selectable_value(&mut app.filter_priority, Some(Priority::Low), "Low");
                    ui.selectable_value(&mut app.filter_priority, Some(Priority::Medium), "Medium");
                    ui.selectable_value(&mut app.filter_priority, Some(Priority::High), "High");
                });

            ui.add(
                egui::TextEdit::singleline(&mut app.search_query)
                    .hint_text("Search items...")
                    .desired_width(if constraints.is_mobile { 130.0 } else { 180.0 }),
            );
        });
    });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(8.0);

    let query = app.search_query.trim().to_lowercase();
    let priority_filter = app.filter_priority;

    let mut item_ids_to_toggle = Vec::new();
    let mut item_ids_to_remove = Vec::new();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let matching_items: Vec<_> = app
                .state
                .collection
                .items
                .iter()
                .filter(|item| {
                    if let Some(pf) = priority_filter
                        && item.priority != pf
                    {
                        return false;
                    }
                    if !query.is_empty() {
                        let match_title = item.title.to_lowercase().contains(&query);
                        let match_desc = item.description.to_lowercase().contains(&query);
                        if !match_title && !match_desc {
                            return false;
                        }
                    }
                    true
                })
                .cloned()
                .collect();

            if matching_items.is_empty() {
                ui.add_space(20.0);
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("No items match your query or filter.").weak());
                });
                return;
            }

            for item in matching_items {
                egui::Frame::group(ui.style())
                    .inner_margin(10.0)
                    .corner_radius(6.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let mut checked = item.completed;
                            if ui.checkbox(&mut checked, "").clicked() {
                                item_ids_to_toggle.push(item.id);
                            }

                            ui.vertical(|ui| {
                                let mut title_text =
                                    egui::RichText::new(&item.title).strong().size(14.0);
                                if item.completed {
                                    title_text = title_text.strikethrough().weak();
                                }
                                ui.label(title_text);

                                if !item.description.is_empty() {
                                    let mut desc_text =
                                        egui::RichText::new(&item.description).small();
                                    if item.completed {
                                        desc_text = desc_text.weak();
                                    }
                                    ui.label(desc_text);
                                }
                            });

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("Delete").on_hover_text("Delete item").clicked() {
                                        item_ids_to_remove.push(item.id);
                                    }

                                    let badge_color = match item.priority {
                                        Priority::High => egui::Color32::from_rgb(220, 70, 70),
                                        Priority::Medium => egui::Color32::from_rgb(230, 140, 50),
                                        Priority::Low => egui::Color32::from_rgb(100, 140, 180),
                                    };
                                    ui.colored_label(
                                        badge_color,
                                        egui::RichText::new(item.priority.label()).strong().small(),
                                    );
                                },
                            );
                        });
                    });
                ui.add_space(4.0);
            }
        });

    let mut collection_changed = false;
    for id in item_ids_to_toggle {
        app.state.collection.toggle(id);
        collection_changed = true;
    }
    for id in item_ids_to_remove {
        app.state.collection.remove(id);
        collection_changed = true;
    }
    if collection_changed {
        app.persist_state();
    }
}
