use crate::config::ParamType;

// PARAM_SIZE has to be VERTEX_COUNT * 2

// configure _ constants below to optimize your graph
// example from https://baehyunsol.github.io/IRs-of-Rust.html
const EDGE_COUNT: usize = 10;
pub const VERTEX_COUNT: usize = 9;
const EDGES: [(usize, usize); EDGE_COUNT] = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 5),
    (5, 6),
    (5, 7),
    (6, 7),
    (6, 8),
];

// it does 2 things:
//    1, prevent the graph from rotating 
//    2, can represent real-world data, if exists
const FIXED_VERTICES: [
    (usize, ParamType, ParamType);  // (index, x, y)
    1   // the number of fixed vertices
] = [
    // let's just fix this vertex at origin
    (0, 0.0, 0.0),
];

// not as good as I thought, though...
// what would be the best?
const MIN_EDGE_SIZE: ParamType = 0.5;
const MAX_EDGE_SIZE: ParamType = 2.0;
const MIN_VERTEX_DISTANCE: ParamType = 0.1;
const FIXED_POINT_TOLERANCE: ParamType = 1.0;

// let's avoid sqrt calls
const MIN_EDGE_SIZE_SQR: ParamType = MIN_EDGE_SIZE * MIN_EDGE_SIZE;
const MAX_EDGE_SIZE_SQR: ParamType = MAX_EDGE_SIZE * MAX_EDGE_SIZE;
const MIN_VERTEX_DISTANCE_SQR: ParamType = MIN_VERTEX_DISTANCE * MIN_VERTEX_DISTANCE;
const FIXED_POINT_TOLERANCE_SQR: ParamType = FIXED_POINT_TOLERANCE * FIXED_POINT_TOLERANCE;

pub fn f(parameters: &[ParamType]) -> ParamType {
    assert_eq!(parameters.len(), VERTEX_COUNT * 2);
    let mut loss: ParamType = 0.0;

    for i in 0..VERTEX_COUNT {
        let x1 = parameters[i * 2];
        let y1 = parameters[i * 2 + 1];

        for j in (i + 1)..VERTEX_COUNT {
            let x2 = parameters[j * 2];
            let y2 = parameters[j * 2 + 1];

            let dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

            if dist < MIN_VERTEX_DISTANCE_SQR {
                if dist == 0.0 {
                    loss += 1e15;  //  a big enough number
                }

                else {
                    loss += 1.0 / dist;
                }
            }
        }
    }

    for (v1, v2) in EDGES.iter() {
        let x1 = parameters[v1 * 2];
        let y1 = parameters[v1 * 2 + 1];

        let x2 = parameters[v2 * 2];
        let y2 = parameters[v2 * 2 + 1];

        let dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

        if dist < MIN_EDGE_SIZE_SQR {
            if dist == 0.0 {
                loss += 1e15;  // a big enough number
            }

            else {
                loss += 1.0 / dist;
            }
        }

        else if dist > MAX_EDGE_SIZE_SQR {
            loss += dist.sqrt();  // this is how force-based drawing works
        }
    }

    for (index, ref_x, ref_y) in FIXED_VERTICES.iter() {
        let curr_x = parameters[index * 2];
        let curr_y = parameters[index * 2 + 1];

        let dist = (curr_x - ref_x) * (curr_x - ref_x) + (curr_y - ref_y) * (curr_y - ref_y);

        if dist > FIXED_POINT_TOLERANCE_SQR {
            loss += dist * 4.0;  // it has to be stronger than other forces ... really?
        }
    }

    loss
}
