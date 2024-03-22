use crate::files::{exists, write_string, FileError, WriteMode};
use chrono::offset::Local;

pub fn initialize_log_file(path: &str) -> Result<(), FileError> {
    if !exists(path) {
        write_string(path, "", WriteMode::AlwaysCreate)?;
    }

    // TODO: do something if the log file is too long

    Ok(())
}

pub fn write_log(
    path: Option<String>,
    owner: &str,
    msg: &str,
) {
    // TODO: do something if the log file is too long

    if let Some(path) = path {
        write_string(
            &path,
            &format!(
                "{:?} | {} | {msg}\n",
                Local::now(),
                if owner.len() < 16 {
                    format!("{}{owner}", " ".repeat(16 - owner.len()))
                } else {
                    owner.to_string()
                },
            ),
            WriteMode::AlwaysAppend,
        ).unwrap();
    }
}
