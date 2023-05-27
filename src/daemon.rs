use crate::state::{State, iterate, init_stepss, init_workers};
use crate::multi::MessageFromMain;
use std::fs::{File, read};
use std::io::Write;
use std::time::Instant;

pub struct Daemon<T> {

    /// Name of the file where its state is saved. If it doesn't exist or is corrupted, it initializes new one.
    pub load_progress_from: String,

    state: State,

    /// Number of parameters it optimizes. It should be less than or equal to 8. If not set, it's 8.
    pub num_params: usize,

    /// `Vec<(x, y)>`
    pub data: Vec<(T, f64)>,

    /// It's a function: `y = f(x)`. The first argument is its parameters and the second one is `x`.
    /// If not set, it's a polynomial (ax^7 + bx^6 + cx^5 + dx^4 + ex^3 + fx^2 + gx + h). Frequently used options are defined in `regression::funcs`.
    pub objective_function: fn(Vec<f64>, T) -> f64,

    /// How often it prints the progress (in milliseconds). If it's 0, it doesn't print anything. If not set, it's 2000.
    pub print_cycle: usize,

    /// Number of processes to spawn. If not set, it's 8.
    pub num_workers: usize,

    /// It ends when the mse gets smaller than this value. If not set, it's 0.
    pub goal_mse: f64,

    /// Doesn't print anything. If not set, it's false.
    pub quiet: bool
}

impl<T: Clone + Send + 'static> Daemon<T> {

    pub fn new(load_progress_from: String, data: Vec<(T, f64)>, objective_function: fn(Vec<f64>, T) -> f64, num_params: usize) -> Self {
        let print_cycle = 2000;
        let num_params = num_params;
        let num_workers = 8;
        let goal_mse = 0.0;
        let quiet = false;

        let state = match read(&load_progress_from) {
            Ok(bytes) => match State::deserialize(&bytes, num_params) {
                Ok(s) => s,
                _ => {
                    if !quiet { println!("The file `{load_progress_from}` is corrupted, creating a default state..."); }
                    State::new(vec![0.0; num_params], vec![2.0; num_params])
                }
            },
            _ => {
                if !quiet { println!("Failed to read the file `{load_progress_from}`, creating a default state..."); }
                State::new(vec![0.0; num_params], vec![2.0; num_params])
            }
        };

        Daemon {
            load_progress_from, num_params, data, objective_function, num_workers, quiet, print_cycle, state, goal_mse
        }
    }

    pub fn init_state(&mut self, params: Vec<f64>, dists: Vec<f64>) {
        self.state = State::new(params, dists);
    }

    pub fn run(&mut self) {
        let stepss = init_stepss(self.num_params);
        let mut channels = init_workers(self.data.clone(), self.objective_function, self.num_workers);
        self.print("Worker Initialization Complete!");

        let mut curr_state = self.state.clone();
        let mut last_print = Instant::now();

        while self.state.mse > self.goal_mse {

            if Instant::now().duration_since(last_print.clone()).as_millis() as usize > self.print_cycle {
                self.print_state();
                self.save_state();
                last_print = Instant::now();
            }

            if let Err(_) = iterate(&mut self.state, &channels, self.num_workers, self.num_params, &stepss) {
                self.print("Error Occured! Trying to restart...");

                for channel in channels.iter() {
                    channel.tx_from_main.send(MessageFromMain::Kill);
                }

                self.state = curr_state;
                channels = init_workers(self.data.clone(), self.objective_function, self.num_workers);
                self.print("Worker Initialization Complete!");
            }

            curr_state = self.state.clone();
        }

        self.print_state();
        self.save_state();
    }

    fn print(&self, msg: &str) {
        if !self.quiet { println!("{msg}"); }
    }

    fn print_state(&self) {
        self.print(&format!("params: {:?}\ndists: {:?}\nepoch: {}, mse: {}", self.state.params, self.state.dists, self.state.epoch, self.state.mse));
    }

    fn save_state(&self) {

        let file_save = match File::create(&self.load_progress_from) {
            Ok(mut f) => f.write_all(&self.state.serialize()).is_ok(),
            _ => { false }
        };

        if !file_save { self.print(&format!("Failed to save the progress to the file: `{}`", self.load_progress_from)); }
    }

}