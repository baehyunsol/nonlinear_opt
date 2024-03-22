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
) {
    (
        8,    // number of parallel workers (it has to be at least 2)
        256,  // iterations per worker
        1.0,  // l2 norm of initial random parameters
        0.9,  // l2 norm of the first step

        // step moment
        // if it's 1, the step is never updated
        // if it's 0, the previous step is completely ignored
        0.1,

        true,  // visualize

        Some(String::from("./log.txt")),  // write logs to here
    )
}

// Function that you're optimizing
pub fn f(parameters: &[ParamType]) -> ParamType {  // returns loss
    // impl body
    todo!()
}