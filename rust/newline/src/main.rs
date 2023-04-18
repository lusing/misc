extern crate gtk;
extern crate gio;

use gtk::prelude::*;
use gio::prelude::*;

use gtk::{Builder, Button, TextView, ApplicationWindow, ScrolledWindow};
use std::env::args;

fn build_ui(application: &gtk::Application) {
    let builder = Builder::from_string(include_str!("remove_newlines.glade"));

    let window: ApplicationWindow = builder.get_object("window").expect("Couldn't get window");
    window.set_application(Some(application));

    let input_scrolled_window: ScrolledWindow = builder
        .get_object("input_scrolled_window")
        .expect("Couldn't get input_scrolled_window");

    let input_text_view: TextView = TextView::new();
    input_scrolled_window.add(&input_text_view);

    let output_scrolled_window: ScrolledWindow = builder
        .get_object("output_scrolled_window")
        .expect("Couldn't get output_scrolled_window");

    let output_text_view: TextView = TextView::new();
    output_scrolled_window.add(&output_text_view);

    let remove_button: Button = builder.get_object("remove_button").expect("Couldn't get remove_button");
    let clear_button: Button = builder.get_object("clear_button").expect("Couldn't get clear_button");
    let copy_button: Button = builder.get_object("copy_button").expect("Couldn't get copy_button");

    remove_button.connect_clicked(move |_| {
        let input_buffer = input_text_view.get_buffer().expect("Couldn't get input buffer");
        let output_buffer = output_text_view.get_buffer().expect("Couldn't get output buffer");
        let input_text = input_buffer.get_text(&input_buffer.get_start_iter(), &input_buffer.get_end_iter(), false).unwrap().to_string();
        let output_text = input_text.replace("\n", " ");
        output_buffer.set_text(&output_text);
    });

    clear_button.connect_clicked(move |_| {
        let input_buffer = input_text_view.get_buffer().expect("Couldn't get input buffer");
        input_buffer.set_text("");
    });

    copy_button.connect_clicked(move |_| {
        let output_buffer = output_text_view.get_buffer().expect("Couldn't get output buffer");
        let output_text = output_buffer.get_text(&output_buffer.get_start_iter(), &output_buffer.get_end_iter(), false).unwrap().to_string();
        if let Some(clipboard) = gtk::Clipboard::get(&gdk::SELECTION_CLIPBOARD) {
            clipboard.set_text(&output_text);
        }
    });

    window.show_all();
}

fn main() {
    let application = gtk::Application::new(
        Some("com.example.remove_newlines"),
        Default::default(),
    )
    .expect("Failed to initialize GTK application");

    application.connect_activate(|app| {
        build_ui(app);
    });

    application.run(&args().collect::<Vec<_>>());
}

