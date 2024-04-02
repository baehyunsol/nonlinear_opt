use crate::config::ParamType;
use crate::utils::get_l2_norm;
use h_time::Date;

// TODO: import/export to file
pub struct State {
    pub id: usize,
    pub parameters: Vec<ParamType>,
    pub prev_step: Option<Vec<ParamType>>,
    pub loss: ParamType,
    pub successful_turns: usize,
    pub failed_turns: usize,
    pub last_updated_at: Option<Date>,
    // TODO: track how loss changes over time (for ex: every 1 minutes)
}

impl State {
    pub fn pretty_print(&self) -> String {
        format!(
            "id: {}\nparameters: {} (l2_norm: {})\n  gradient: {}\n      loss: {}\nsuccessful turns: {}\nfailed turns: {}\n{}",
            self.id,
            pretty_print_vec_float(&self.parameters, false),
            get_l2_norm(&self.parameters),
            self.prev_step.as_ref().map(
                |s| format!(
                    "{} (l2_norm: {})",
                    pretty_print_vec_float(s, false),
                    get_l2_norm(s),
                )
            ).unwrap_or_else(|| String::from("None")),
            self.loss,
            self.successful_turns,
            self.failed_turns,
            if let Some(t) = &self.last_updated_at {
                format!("last updated {} seconds ago", Date::now().duration_since(&t).into_secs())
            } else {
                String::new()
            },
        )
    }
}

fn pretty_print_vec_float(v: &[ParamType], show_dots: bool) -> String {
    if v.len() > 8 {
        pretty_print_vec_float(&v[..8], true)
    }

    else {
        let ss = v.iter().map(|p| pretty_print_float(*p)).collect::<Vec<_>>().join(",");

        format!(
            "[{ss}{}]",
            if show_dots {
                "..."
            } else {
                ""
            },
        )
    }
}

fn pretty_print_float(f: ParamType) -> String {
    let s = format!("{f:.4}");

    if s.len() < 9 {
        format!(
            "{}{s}",
            " ".repeat(9 - s.len()),
        )
    }

    else {
        s
    }
}
