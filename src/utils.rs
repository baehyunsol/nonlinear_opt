use crate::config::ParamType;

pub fn generate_random_params(length: usize, l2_norm: ParamType) -> Vec<ParamType> {
    let mut result = (0..length).map(|_| rand::random::<ParamType>() - 0.5).collect::<Vec<_>>();
    let curr_l2_norm = get_l2_norm(&result);

    mul_k_params(&mut result, l2_norm / curr_l2_norm);

    result
}

pub fn get_l2_norm(params: &[ParamType]) -> ParamType {
    let sum = params.iter().map(|p| p * p).sum::<ParamType>();

    sum.sqrt()
}

pub fn get_distance_of_params(p1: &[ParamType], p2: &[ParamType]) -> ParamType {
    assert_eq!(p1.len(), p2.len());
    let mut sum = 0.0;

    for i in 0..p1.len() {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }

    sum.sqrt()
}

pub fn mul_k_params(params: &mut Vec<ParamType>, k: ParamType) {
    for p in params.iter_mut() {
        *p *= k;
    }
}

pub fn add_params(params: &mut Vec<ParamType>, val: &[ParamType]) {
    assert_eq!(params.len(), val.len(), "cannot add 2 vectors with different lengths");

    for i in 0..params.len() {
        params[i] += val[i];
    }
}

pub fn sub_params(params: &mut Vec<ParamType>, val: &[ParamType]) {
    assert_eq!(params.len(), val.len(), "cannot subtract 2 vectors with different lengths");

    for i in 0..params.len() {
        params[i] -= val[i];
    }
}
