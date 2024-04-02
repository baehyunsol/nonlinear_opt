// either `f32` or `f64`
pub type ParamType = f32;

// parameters.len()
pub const PARAM_SIZE: usize = 32;

// make sure that `f` never returns a value greater than this
pub const VERY_BIG_LOSS: ParamType = 3e20;

// modify this function to change configs
pub fn default_config() -> (
    usize,
    usize,
    ParamType,
    ParamType,
    ParamType,
    bool,
    Option<String>,
    bool,
) {
    (
        8,    // number of parallel workers (it has to be at least 2)
        512,  // iterations per worker
        1.0,  // l2 norm of initial random parameters
        0.5,  // l2 norm of the first step

        // step moment
        // if it's 1, the step is never updated
        // if it's 0, the previous step is completely ignored
        0.65,

        true,  // visualize

        Some(String::from("./log.txt")),  // write logs to here
        true,  // remove existing log file
    )
}

// Function that you're optimizing
pub fn f(parameters: &[ParamType]) -> ParamType {  // returns loss
    // impl body
    todo!()
}

// dependencies of the default visualizer
use crate::state::State;
use std::time::Instant;

pub fn visualizer(states: &[State]) {
    clearscreen::clear().unwrap();

    for state in states.iter() {
        println!("\n{}", state.pretty_print());
    }
}
