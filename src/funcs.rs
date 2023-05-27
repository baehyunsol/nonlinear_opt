//! Predefined objective functions

/// y = ax^2 + bx + c\
/// params = [a, b, c]
pub fn quad(params: Vec<f64>, x: f64) -> f64 {
    params[0] * x * x + params[1] * x + params[2]
}

/// y = ax^7 + bx^6 + cx^5 + dx^4 + ex^3 + fx^2 + gx + h\
/// params: [a, b, c, d, e, f, g, h]
pub fn poly(params: Vec<f64>, x: f64) -> f64 {
    params[0] * x.powi(7) + params[1] * x.powi(6) + params[2] * x.powi(5) + params[3] * x.powi(4) + params[4] * x.powi(3) + params[5] * x * x + params[6] * x + params[7]
}

/// y = a * sin(bx + c) + d * sin(ex + f) + g\
/// params: [a, b, c, d, e, f, g]
pub fn trigo(params: Vec<f64>, x: f64) -> f64 {
    params[0] * (params[1] * x + params[2]).sin() + params[3] * (params[4] * x + params[5]).sin() + params[6]
}

/// y = ax1 + bx2 + cx3 + dx4 + ex5 + fx6 + gx7 + h\
/// params: [a, b, c, d, e, f, g, h]\
/// x: [x1, x2, x3, x4, x5, x6, x7]
pub fn linear(params: Vec<f64>, x: [f64; 7]) -> f64 {
    params[0] * x[0] + params[1] * x[1] + params[2] * x[2] + params[3] * x[3] + params[4] * x[4] + params[5] * x[5] + params[6] * x[6] + params[7]
}

/// y = exp(ax + b) + exp(cx + d) + exp(ex + f) + g\
/// params: [a, b, c, d, e, f, g]
pub fn exp(params: Vec<f64>, x: f64) -> f64 {
    (params[0] * x + params[1]).exp() + (params[2] * x + params[3]).exp() + (params[4] * x + params[5]).exp() + params[6]
}