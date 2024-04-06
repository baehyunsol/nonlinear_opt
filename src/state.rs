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
    pub losses_over_time: Vec<(Date, ParamType)>,
}

impl State {
    pub fn update_best_loss(
        &mut self,
        new_params: Vec<ParamType>,
        new_loss: ParamType,
        prev_step: Vec<ParamType>,
    ) {
        let now = Date::now();

        self.parameters = new_params;
        self.loss = new_loss;
        self.prev_step = Some(prev_step);
        self.last_updated_at = Some(now.clone());
        self.successful_turns += 1;

        if self.losses_over_time.len() < 64 {
            self.losses_over_time.push((now, new_loss));
        }

        else {
            let last_check_point = self.losses_over_time.pop().unwrap().0;

            if now.duration_since(&last_check_point).into_minutes() > 3 {
                self.losses_over_time.push((now, new_loss));
                self.losses_over_time = self.losses_over_time[1..].to_vec();
            }
        }
    }

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
