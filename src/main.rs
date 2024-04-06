mod config;
mod files;
mod log;
mod multi;
mod samples;
mod state;
mod utils;

use h_time::Date;
use log::{initialize_log_file, write_log};
use multi::{
    init_channels,
    MessageFromMain,
    MessageToMain,
};
use state::State;
use std::thread;
use std::time::Duration;

fn main() {
    let (
        num_workers,
        iter_per_worker,
        initial_l2_norm,
        initial_step_size,
        step_moment,
        visualize,
        write_logs_to,
        remove_existing_log_file,
    ) = config::default_config();

    if let Some(path) = &write_logs_to {
        initialize_log_file(path, remove_existing_log_file).unwrap();
    }

    write_log(
        write_logs_to.clone(),
        "master",
        "hello from master",
    );

    if num_workers < 2 {
        let error_message = "num_workers has to be at least 2! aborting...";

        write_log(
            write_logs_to.clone(),
            "master",
            error_message,
        );

        panic!("{error_message}");
    }

    if step_moment < 0.0 || step_moment > 1.0 {
        let error_message = "step_moment has to be 0 ~ 1";

        write_log(
            write_logs_to.clone(),
            "master",
            error_message,
        );

        panic!("{error_message}");
    }

    let channels = multi::init_channels(
        num_workers,
    );

    for channel in channels.iter() {
        channel.send(MessageFromMain::TryRandomParams {
            param_l2_norm: initial_l2_norm,
            count: iter_per_worker,
        }).unwrap();
    }

    let mut good_random_params = vec![];

    // waits until the workers finish trying random params
    while good_random_params.len() < channels.len() {
        for channel in channels.iter() {
            if let Ok(msg) = channel.try_recv() {
                match msg {
                    MessageToMain::RandomParamResult {
                        best_params,
                        best_loss,
                    } => {
                        write_log(
                            write_logs_to.clone(),
                            "master",
                            &format!("got message: random_param_result(loss: {best_loss:.4})"),
                        );
                        good_random_params.push((best_params, best_loss));
                    },
                    _ => unreachable!(),
                }
            }

            if let Err(_) = channel.send(MessageFromMain::HealthCheck) {
                // TODO: revive
                write_log(
                    write_logs_to.clone(),
                    "master",
                    "found a dead worker",
                );
            }
        }

        // no need to run a busy loop
        thread::sleep(Duration::from_millis(500));
    }

    let mut distances = vec![];

    for i in 0..good_random_params.len() {
        for j in (i + 1)..good_random_params.len() {
            distances.push((i, j, utils::get_distance_of_params(&good_random_params[i].0, &good_random_params[j].0)));
        }
    }

    distances.sort_by(|(_, _, dist1), (_, _, dist2)| dist1.partial_cmp(dist2).unwrap());

    let now = Date::now();
    let mut states = vec![
        State {
            id: 0,
            parameters: good_random_params[distances.last().unwrap().0].0.clone(),
            prev_step: None,
            loss: good_random_params[distances.last().unwrap().0].1,
            successful_turns: 0,
            failed_turns: 0,
            last_updated_at: Some(now.clone()),
            losses_over_time: vec![(now, good_random_params[distances.last().unwrap().0].1)],
        },
        State {
            id: 1,
            parameters: good_random_params[distances.last().unwrap().1].0.clone(),
            prev_step: None,
            loss: good_random_params[distances.last().unwrap().1].1,
            successful_turns: 0,
            failed_turns: 0,
            last_updated_at: Some(now.clone()),
            losses_over_time: vec![(now, good_random_params[distances.last().unwrap().1].1)],
        },
    ];

    for channel in channels.iter() {
        for state in states.iter() {
            channel.send(MessageFromMain::TryWithGradient {
                state_id: state.id,
                curr_params: state.parameters.clone(),
                prev_step: state.prev_step.clone(),
                step_size: initial_step_size,
                step_moment,
                count: iter_per_worker,
            }).unwrap();
        }
    }

    loop {
        for channel in channels.iter() {
            while let Ok(msg) = channel.try_recv() {
                match msg {
                    MessageToMain::WithGradientResult {
                        state_id,
                        best_params,
                        best_loss,
                        step,
                    } => {
                        write_log(
                            write_logs_to.clone(),
                            "master",
                            &format!("got message: with_gradient_result(state: {state_id}, loss: {best_loss:.4})"),
                        );

                        if best_loss < states[state_id].loss {
                            states[state_id].update_best_loss(
                                best_params.clone(),
                                best_loss,
                                step.clone(),
                            );
                        }

                        else {
                            states[state_id].failed_turns += 1;
                        }

                        if let Err(_) = channel.send(MessageFromMain::TryWithGradient {
                            state_id: state_id,
                            curr_params: best_params,
                            prev_step: Some(step),
                            step_size: initial_step_size,
                            step_moment,
                            count: iter_per_worker,
                        }) {
                            // TODO: revive this channel
                        }
                    },
                    MessageToMain::WithGradientResultFailure { state_id } => {
                        write_log(
                            write_logs_to.clone(),
                            "master",
                            &format!("got message: with_gradient_result_failure(state: {state_id})"),
                        );

                        states[state_id].failed_turns += 1;

                        if let Err(_) = channel.send(MessageFromMain::TryWithGradient {
                            state_id: state_id,
                            curr_params: states[state_id].parameters.clone(),
                            prev_step: states[state_id].prev_step.clone(),
                            step_size: initial_step_size,
                            step_moment,
                            count: iter_per_worker,
                        }) {
                            // TODO: revive this channel
                        }
                    },
                    MessageToMain::RandomParamResult { .. } => unreachable!(),
                }
            }
        }

        if visualize {
            config::visualizer(&states);
        }

        // no need to run a busy loop
        thread::sleep(Duration::from_millis(800));
    }
}
