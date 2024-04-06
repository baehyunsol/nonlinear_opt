use crate::config::ParamType;

const INPUT_DIMENSION: usize = 256;
const HIDDEN_STATE_DIMENSION: usize = 1024;
const OUTPUT_DIMENSION: usize = 256;

const PARAM_SIZE: usize = HIDDEN_STATE_DIMENSION * 2 * OUTPUT_DIMENSION + OUTPUT_DIMENSION  // calc_output
    + (HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION) * 4;  // f_t, i_t, o_t, g_t

// TODO: use `get_unchecked` (when you're sure that the code is robust enough)

pub fn time_step(
    parameters: &[ParamType],
    cell_state: &[ParamType],    // c_(t-1)
    hidden_state: &[ParamType],  // h_(t-1)
    input: &[ParamType],   // x_t
    output: bool,
) -> (
    Vec<ParamType>,  // next_cell_state
    Vec<ParamType>,  // next_hidden_state
    Option<Vec<ParamType>>,  // output
) {
    debug_assert_eq!(parameters.len(), PARAM_SIZE);
    debug_assert_eq!(cell_state.len(), HIDDEN_STATE_DIMENSION);
    debug_assert_eq!(hidden_state.len(), HIDDEN_STATE_DIMENSION);
    debug_assert_eq!(input.len(), INPUT_DIMENSION);

    let concat_input = vec![
        hidden_state.to_vec(),
        input.to_vec(),
    ].concat();
    let mut offset = 0;

    let forget_gate = calc_forget_gate(
        &parameters[offset..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))],
        &concat_input,
        &parameters[(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION)],
    );
    offset += HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION;

    let input_gate = calc_input_gate(
        &parameters[offset..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))],
        &concat_input,
        &parameters[(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION)],
    );
    offset += HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION;

    let output_gate = calc_output_gate(
        &parameters[offset..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))],
        &concat_input,
        &parameters[(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION)],
    );
    offset += HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION;

    let activation = calc_activation(
        &parameters[offset..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))],
        &concat_input,
        &parameters[(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION))..(offset + HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION)],
    );
    offset += HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + HIDDEN_STATE_DIMENSION;

    let next_cell_state = calc_next_cell_state(
        &forget_gate,
        cell_state,
        &input_gate,
        &activation,
    );

    let next_hidden_state = calc_hidden_state(
        &output_gate,
        &next_cell_state,
    );

    let output = if output {
        let concat_state = vec![
            next_cell_state.clone(),
            next_hidden_state.clone(),
        ].concat();

        Some(calc_output(
            &parameters[offset..(offset + HIDDEN_STATE_DIMENSION * 2 * OUTPUT_DIMENSION)],
            &concat_state,
            &parameters[(offset + HIDDEN_STATE_DIMENSION * 2 * OUTPUT_DIMENSION)..],
        ))
    } else {
        None
    };

    (next_cell_state, next_hidden_state, output)
}

pub fn one_hot_encode(byte: u8) -> Vec<ParamType> {
    assert_eq!(INPUT_DIMENSION, 256);
    let mut result = vec![0.0; 256];
    result[byte as usize] = 1.0;

    result
}

pub fn one_hot_decode(token: &[ParamType]) -> u8 {
    assert_eq!(token.len(), 256);
    let mut curr_max = ParamType::MIN;
    let mut curr_max_index = 0;

    for (index, n) in token.iter().enumerate() {
        if *n > curr_max {
            curr_max = *n;
            curr_max_index = index;
        }
    }

    curr_max_index as u8
}

// f_t: HIDDEN_STATE_DIMENSION
// i_t: HIDDEN_STATE_DIMENSION
// o_t: HIDDEN_STATE_DIMENSION
// g_t: HIDDEN_STATE_DIMENSION
// c_t: HIDDEN_STATE_DIMENSION
// h_t: HIDDEN_STATE_DIMENSION

// not in the original LSTM design, only in my own version
// Y = σ(W_y × (c_t <> h_t) + b_y)
fn calc_output(
    output_weight: &[ParamType],  // W_y
    concat_state: &[ParamType],   // c_t <> h_t
    bias: &[ParamType],           // b_y
) -> Vec<ParamType> {
    debug_assert_eq!(output_weight.len(), HIDDEN_STATE_DIMENSION * 2 * OUTPUT_DIMENSION);
    debug_assert_eq!(concat_state.len(), HIDDEN_STATE_DIMENSION * 2);
    debug_assert_eq!(bias.len(), OUTPUT_DIMENSION);

    (0..OUTPUT_DIMENSION).map(
        |index1| sigmoid((0..(HIDDEN_STATE_DIMENSION * 2)).map(
            |index2| output_weight[index1 * (HIDDEN_STATE_DIMENSION * 2) + index2] * concat_state[index2]
        ).sum::<ParamType>() + bias[index1])
    ).collect()
}

// h_t = o_t * tanh(c_t)
fn calc_hidden_state(
    output_gate: &[ParamType],  // o_t
    cell_state: &[ParamType],   // c_t
) -> Vec<ParamType> {
    debug_assert_eq!(output_gate.len(), HIDDEN_STATE_DIMENSION);
    debug_assert_eq!(cell_state.len(), HIDDEN_STATE_DIMENSION);

    (0..HIDDEN_STATE_DIMENSION).map(
        |index| output_gate[index] * cell_state[index].tanh()
    ).collect()
}

// c_t = f_t * c_(t-1) + i_t * g_t
fn calc_next_cell_state(
    forget_gate: &[ParamType],  // f_t
    previous_cell_state: &[ParamType],  // c_(t-1)
    input_gate: &[ParamType],   // i_t
    activation: &[ParamType],   // g_t
) -> Vec<ParamType> {
    debug_assert_eq!(forget_gate.len(), HIDDEN_STATE_DIMENSION);
    debug_assert_eq!(previous_cell_state.len(), HIDDEN_STATE_DIMENSION);
    debug_assert_eq!(input_gate.len(), HIDDEN_STATE_DIMENSION);
    debug_assert_eq!(activation.len(), HIDDEN_STATE_DIMENSION);

    (0..HIDDEN_STATE_DIMENSION).map(
        |index| forget_gate[index] * previous_cell_state[index] + input_gate[index] * activation[index]
    ).collect()
}

// f_t = σ(W_f × (h_(t-1) <> x_t) + b_f)
fn calc_forget_gate(
    forget_gate_weight: &[ParamType],  // W_f
    concat_input: &[ParamType],  // h_(t-1) <> x_t
    bias: &[ParamType],  // b_f
) -> Vec<ParamType> {
    debug_assert_eq!(forget_gate_weight.len(), HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION));
    debug_assert_eq!(concat_input.len(), HIDDEN_STATE_DIMENSION + INPUT_DIMENSION);
    debug_assert_eq!(bias.len(), HIDDEN_STATE_DIMENSION);

    (0..HIDDEN_STATE_DIMENSION).map(
        |index1| sigmoid((0..(HIDDEN_STATE_DIMENSION + INPUT_DIMENSION)).map(
            |index2| forget_gate_weight[index1 * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + index2] * concat_input[index2]
        ).sum::<ParamType>() + bias[index1])
    ).collect()
}

// i_t = σ(W_i × (h_(t-1) <> x_t) + b_i)
fn calc_input_gate(
    input_gate_weight: &[ParamType],  // W_i
    concat_input: &[ParamType],  // h_(t-1) <> x_t
    bias: &[ParamType],  // b_i
) -> Vec<ParamType> {
    debug_assert_eq!(input_gate_weight.len(), HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION));
    debug_assert_eq!(concat_input.len(), HIDDEN_STATE_DIMENSION + INPUT_DIMENSION);
    debug_assert_eq!(bias.len(), HIDDEN_STATE_DIMENSION);

    (0..HIDDEN_STATE_DIMENSION).map(
        |index1| sigmoid((0..(HIDDEN_STATE_DIMENSION + INPUT_DIMENSION)).map(
            |index2| input_gate_weight[index1 * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + index2] * concat_input[index2]
        ).sum::<ParamType>() + bias[index1])
    ).collect()
}

// o_t = σ(W_i × (h_(t-1) <> x_t) + b_o)
fn calc_output_gate(
    output_gate_weight: &[ParamType],  // W_o
    concat_input: &[ParamType],  // h_(t-1) <> x_t
    bias: &[ParamType],  // b_o
) -> Vec<ParamType> {
    debug_assert_eq!(output_gate_weight.len(), HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION));
    debug_assert_eq!(concat_input.len(), HIDDEN_STATE_DIMENSION + INPUT_DIMENSION);
    debug_assert_eq!(bias.len(), HIDDEN_STATE_DIMENSION);

    (0..HIDDEN_STATE_DIMENSION).map(
        |index1| sigmoid((0..(HIDDEN_STATE_DIMENSION + INPUT_DIMENSION)).map(
            |index2| output_gate_weight[index1 * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + index2] * concat_input[index2]
        ).sum::<ParamType>() + bias[index1])
    ).collect()
}

// g_t = σ(W_g × (h_(t-1) <> x_t) + b_g)
fn calc_activation(
    activation_weight: &[ParamType],  // W_g
    concat_input: &[ParamType],       // h_(t-1) <> x_t
    bias: &[ParamType],  // b_g
) -> Vec<ParamType> {
    debug_assert_eq!(activation_weight.len(), HIDDEN_STATE_DIMENSION * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION));
    debug_assert_eq!(concat_input.len(), HIDDEN_STATE_DIMENSION + INPUT_DIMENSION);
    debug_assert_eq!(bias.len(), HIDDEN_STATE_DIMENSION);

    (0..HIDDEN_STATE_DIMENSION).map(
        |index1| sigmoid((0..(HIDDEN_STATE_DIMENSION + INPUT_DIMENSION)).map(
            |index2| activation_weight[index1 * (HIDDEN_STATE_DIMENSION + INPUT_DIMENSION) + index2] * concat_input[index2]
        ).sum::<ParamType>() + bias[index1])
    ).collect()
}

fn sigmoid(x: ParamType) -> ParamType {
    1.0 / (1.0 + (-x).exp())
}
