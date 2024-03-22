use crate::config::{
    default_config,
    f,
    ParamType,
    PARAM_SIZE,
    VERY_BIG_LOSS,
};
use crate::log::write_log;
use crate::utils::{
    add_params,
    generate_random_params,
    get_l2_norm,
    mul_k_params,
    sub_params,
};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

pub enum MessageFromMain {
    TryRandomParams {
        param_l2_norm: ParamType,
        count: usize,
    },
    TryWithGradient {
        state_id: usize,
        curr_params: Vec<ParamType>,
        prev_step: Option<Vec<ParamType>>,

        // a value between 0 ~ 1
        // new_step = prev_step * moment + rand * (1 - moment)
        step_moment: ParamType,

        // this is used only when `prev_step` is None
        step_size: ParamType,
        count: usize,
    },

    // There's no need to respond to this message.
    // failure of `.send(HealthCheck).unwrap()` means the other end is dead,
    // but the success of `.send(HealthCheck).unwrap()` does not guarantee that the other end is alive
    HealthCheck,
}

pub enum MessageToMain {
    RandomParamResult {
        best_params: Vec<ParamType>,
        best_loss: ParamType,
    },
    WithGradientResult {
        state_id: usize,
        best_params: Vec<ParamType>,
        best_loss: ParamType,

        // best - previous
        step: Vec<ParamType>,
    },
    WithGradientResultFailure {
        state_id: usize,
    },
}

pub struct Channel {
    tx_from_main: mpsc::Sender<MessageFromMain>,
    rx_to_main: mpsc::Receiver<MessageToMain>,
}

impl Channel {
    pub fn send(&self, msg: MessageFromMain) -> Result<(), mpsc::SendError<MessageFromMain>> {
        self.tx_from_main.send(msg)
    }

    pub fn try_recv(&self) -> Result<MessageToMain, mpsc::TryRecvError> {
        self.rx_to_main.try_recv()
    }

    pub fn block_recv(&self) -> Result<MessageToMain, mpsc::RecvError> {
        self.rx_to_main.recv()
    }
}

pub fn init_channels(n: usize) -> Vec<Channel> {
    (0..n).map(|_| init_channel()).collect()
}

pub fn init_channel() -> Channel {
    let (tx_to_main, rx_to_main) = mpsc::channel();
    let (tx_from_main, rx_from_main) = mpsc::channel();

    thread::spawn(move || {
        event_loop(tx_to_main, rx_from_main);
    });

    Channel {
        rx_to_main, tx_from_main
    }
}

pub fn distribute_messages(
    messages: Vec<MessageFromMain>,
    channels: &[Channel],
) -> Result<(), mpsc::SendError<MessageFromMain>> {
    for (index, message) in messages.into_iter().enumerate() {
        channels[index % channels.len()].send(message)?;
    }

    Ok(())
}

pub fn event_loop(tx_to_main: mpsc::Sender<MessageToMain>, rx_from_main: mpsc::Receiver<MessageFromMain>) {
    let worker_id = rand::random::<u32>() & 0xfff_ffff;
    let worker_name = format!("worker-{worker_id:x}");
    let (_, _, _, _, _, _, write_logs_to) = default_config();

    write_log(
        write_logs_to.clone(),
        &worker_name,
        &format!("hello from {worker_name}"),
    );

    loop {
        while let Ok(msg) = rx_from_main.try_recv() {
            match msg {
                MessageFromMain::TryRandomParams {
                    count,
                    param_l2_norm,
                } => {
                    write_log(
                        write_logs_to.clone(),
                        &worker_name,
                        &format!("got message: try_random_params"),
                    );
                    let mut curr_best_params = generate_random_params(
                        PARAM_SIZE,
                        param_l2_norm,
                    );
                    let mut curr_best_loss = f(&curr_best_params);

                    for _ in 0..(count - 1) {
                        let new_params = generate_random_params(
                            PARAM_SIZE,
                            param_l2_norm,
                        );
                        let new_loss = f(&new_params);

                        if new_loss < curr_best_loss {
                            curr_best_params = new_params;
                            curr_best_loss = new_loss;
                        }
                    }

                    tx_to_main.send(MessageToMain::RandomParamResult {
                        best_params: curr_best_params,
                        best_loss: curr_best_loss,
                    }).unwrap();
                },
                MessageFromMain::TryWithGradient {
                    state_id,
                    curr_params,
                    prev_step: Some(prev_step),
                    step_moment,
                    step_size: _,
                    count,
                } => {
                    write_log(
                        write_logs_to.clone(),
                        &worker_name,
                        &format!("got message: try_with_gradient(prev_step: Some(...))"),
                    );
                    assert!(0.0 <= step_moment && step_moment <= 1.0);

                    let prev_step_size = get_l2_norm(&prev_step);

                    // new step = weighted_prev_step + rand
                    let mut weighted_prev_step = prev_step.clone();
                    mul_k_params(&mut weighted_prev_step, step_moment);

                    let rand_step_size = (1.0 - step_moment) * prev_step_size;

                    let mut curr_best_loss = VERY_BIG_LOSS;
                    let mut curr_best_params = curr_params.clone();

                    for _ in 0..count {
                        let d_step = generate_random_params(
                            PARAM_SIZE,
                            rand_step_size,
                        );

                        let mut new_step = weighted_prev_step.clone();
                        add_params(&mut new_step, &d_step);

                        let new_step_size = get_l2_norm(&new_step);
                        mul_k_params(&mut new_step, prev_step_size / new_step_size);

                        let mut new_params = curr_params.clone();
                        add_params(&mut new_params, &new_step);

                        let new_loss = f(&new_params);

                        if new_loss < curr_best_loss {
                            curr_best_loss = new_loss;
                            curr_best_params = new_params;
                        }
                    }

                    if curr_best_params == curr_params {
                        tx_to_main.send(MessageToMain::WithGradientResultFailure { state_id }).unwrap();
                    }

                    else {
                        let mut calc_step = curr_best_params.clone();
                        sub_params(&mut calc_step, &curr_params);

                        tx_to_main.send(MessageToMain::WithGradientResult {
                            state_id,
                            best_params: curr_best_params,
                            best_loss: curr_best_loss,
                            step: calc_step,
                        }).unwrap();
                    }
                },
                MessageFromMain::TryWithGradient {
                    state_id,
                    curr_params,
                    prev_step: None,
                    step_moment: _,
                    step_size,
                    count,
                } => {
                    write_log(
                        write_logs_to.clone(),
                        &worker_name,
                        &format!("got message: try_with_gradient(prev_step: None)"),
                    );

                    let mut curr_best_params = curr_params.clone();
                    let mut curr_best_loss = VERY_BIG_LOSS;

                    for _ in 0..count {
                        let new_step = generate_random_params(
                            PARAM_SIZE,
                            step_size,
                        );

                        let mut new_params = curr_params.clone();
                        add_params(&mut new_params, &new_step);

                        let new_loss = f(&new_params);

                        if new_loss < curr_best_loss {
                            curr_best_loss = new_loss;
                            curr_best_params = new_params;
                        }
                    }

                    if curr_best_params == curr_params {
                        tx_to_main.send(MessageToMain::WithGradientResultFailure { state_id }).unwrap();
                    }

                    else {
                        let mut calc_step = curr_best_params.clone();
                        sub_params(&mut calc_step, &curr_params);

                        tx_to_main.send(MessageToMain::WithGradientResult {
                            state_id,
                            best_params: curr_best_params,
                            best_loss: curr_best_loss,
                            step: calc_step,
                        }).unwrap();
                    }
                },
                MessageFromMain::HealthCheck => {
                    write_log(
                        write_logs_to.clone(),
                        &worker_name,
                        &format!("got message: health_check"),
                    );
                },
            }
        }

        // there's no need to run a busy loop
        thread::sleep(Duration::from_millis(3000));
    }
}
