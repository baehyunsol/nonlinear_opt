use crate::config::ParamType;

// TODO
// how about a byte level LSTM?
// input: one-hot encoded bytes
// hidden state: arbitrary size (let's say 1024 for now)
// output: one-hot encoded bytes, there's a matrix that maps the hidden state to the output (256 * 1024 matrix)
// in the original implementation, h_t is an output. but i want another matrix that maps h_t to an actual output

const TOKEN_DIMENSION: usize = 256;
const HIDDEN_STATE_DIMENSION: usize = 1024;
const OUTPUT_DIMENSION: usize = 256;

// file:///home/baehyunsol/Downloads/laptop/240117/Documents/SNU/DL/1114.html

// f_t: HIDDEN_STATE_DIMENSION
// c_t: HIDDEN_STATE_DIMENSION
// i_t: HIDDEN_STATE_DIMENSION
// g_t: HIDDEN_STATE_DIMENSION

// c_t = f_t * c_(t-1) + i_t * g_t
fn get_next_cell_state(
    forget_gate: &[ParamType],  // f_t
    previous_cell_state: &[ParamType],  // c_(t-1)
    input_gate: &[ParamType],   // i_t
    activation: &[ParamType],   // g_t
) -> Vec<ParamType> {}

// f_t = σ(W_f × (h_(t-1) <> x_t) + b_f)
fn calc_forget_gate(
    forget_gate_weight: &[ParamType],  // W_f
    concat_input: &[ParamType],  // h_(t-1) <> x_t
    bias: &[ParamType],  // b_f
) -> Vec<ParamType> {}
