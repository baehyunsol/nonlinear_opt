use std::sync::mpsc;
use std::thread;

pub enum MessageFromMain<T> {
    Initialize {
        data: Vec<(T, f64)>,
        func: fn(Vec<f64>, T) -> f64,
    },
    GetMSE {
        params: Vec<f64>,
        dists: Vec<f64>,
        steps: Vec<i32>
    },
    Kill
}

pub enum MessageToMain {
    NewMSE {
        params: Vec<f64>,
        steps: Vec<i32>,
        mse: f64
    }
}

pub struct Channel<T> {
    pub tx_from_main: mpsc::Sender<MessageFromMain<T>>,
    pub rx_to_main: mpsc::Receiver<MessageToMain>,
}

pub fn init_loop<T: Send + Clone + 'static>() -> Channel<T> {
    let (tx_to_main, rx_to_main) = mpsc::channel();
    let (tx_from_main, rx_from_main) = mpsc::channel();

    thread::spawn(move || {
        event_loop(tx_to_main, rx_from_main);
    });

    Channel {
        rx_to_main, tx_from_main
    }

}

pub fn event_loop<T: Clone>(tx_to_main: mpsc::Sender<MessageToMain>, rx_from_main: mpsc::Receiver<MessageFromMain<T>>) {
    let mut data = vec![];
    let mut func = dummy as fn(Vec<f64>, T) -> f64;

    for msg in rx_from_main {

        match msg {
            MessageFromMain::Initialize { data: data_, func: func_ } => {
                data = data_;
                func = func_;
            }
            MessageFromMain::GetMSE { mut params, dists, steps } => {
                let mut se = 0.0;

                for i in 0..params.len() {
                    params[i] += dists[i] * (steps[i] as f64);
                }

                for (x, y) in data.iter() {
                    let y_ = func(params.clone(), x.clone());
                    se += (*y - y_) * (*y - y_);
                }

                tx_to_main.send(MessageToMain::NewMSE { mse: se, params, steps }).unwrap();
            }
            MessageFromMain::Kill => {
                break;
            }
        }

    }

}

fn dummy<T>(_: Vec<f64>, _: T) -> f64 {
    panic!()
}