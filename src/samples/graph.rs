use crate::config::ParamType;

// This sample is a graph optimizer based on [force-directed graph drawing](https://en.wikipedia.org/wiki/Force-directed_graph_drawing).
// This sample is to test the optimizer, not the graph optimizer. If you want a graph drawer, just use graphviz.

// PARAM_SIZE has to be VERTEX_COUNT * 2

// configure 3 constants below to optimize your graph
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

const EDGE_LENGTH: ParamType = 1.0;

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

            if dist == 0.0 {
                loss += 1e15;  //  a big enough number
            }

            else {
                loss += 1.0 / dist;
            }
        }
    }

    for (v1, v2) in EDGES.iter() {
        let x1 = parameters[v1 * 2];
        let y1 = parameters[v1 * 2 + 1];

        let x2 = parameters[v2 * 2];
        let y2 = parameters[v2 * 2 + 1];

        let dist = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)).sqrt();
        let force = (dist - EDGE_LENGTH).abs();
        loss += force;
    }

    for (index, ref_x, ref_y) in FIXED_VERTICES.iter() {
        let curr_x = parameters[index * 2];
        let curr_y = parameters[index * 2 + 1];

        let dist = (curr_x - ref_x) * (curr_x - ref_x) + (curr_y - ref_y) * (curr_y - ref_y);

        loss += dist;  // it has to be stronger than other forces ... really?
    }

    loss
}

use crate::state::State;
use crate::files::write_string;

pub fn visualizer(states: &[State]) {
    clearscreen::clear().unwrap();

    for state in states.iter() {
        println!("\n{}", state.pretty_print());
    }

    // I don't want to introduce another dependency for this sample
    for (index, state) in states.iter().enumerate() {
        let python = format!("
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 800))

parameters = {:?}
edges = {EDGES:?}

ZOOM = 120
OFFSET = 400

for edge in edges:
    index1, index2 = edge
    x1 = parameters[index1 * 2] * ZOOM + OFFSET
    y1 = parameters[index1 * 2 + 1] * ZOOM + OFFSET
    x2 = parameters[index2 * 2] * ZOOM + OFFSET
    y2 = parameters[index2 * 2 + 1] * ZOOM + OFFSET

    pygame.draw.line(screen, (255, 0, 255), (x1, y1), (x2, y2), 2)

for i in range(len(parameters) // 2):
    x = parameters[i * 2] * ZOOM + OFFSET
    y = parameters[i * 2 + 1] * ZOOM + OFFSET

    pygame.draw.circle(screen, (255, 255, 255), (x, y), 6, 0)

pygame.display.update()
clock = pygame.time.Clock()

while True:
    clock.tick(45)
    pygame.display.update()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            quit()
",
        state.parameters,
        );

        write_string(
            &format!("./draw{index}.py"),
            &python,
            crate::files::WriteMode::CreateOrTruncate,
        ).unwrap();
    }
}
