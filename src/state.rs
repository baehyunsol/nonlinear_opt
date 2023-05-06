use crate::err::*;
use crate::multi::*;
use crate::consts::*;

#[derive(Clone, Debug)]
pub struct State {
    pub params: [f64; 8],
    pub dists: [f64; 8],
    pub mse: f64,
    pub epoch: usize
}

impl State {

    pub fn new(params: [f64; 8], dists: [f64; 8]) -> Self {
        State {
            params, dists,
            mse: f64::MAX,
            epoch: 0
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(144);

        for p in self.params.iter() {

            for byte in u64_to_u8s(f64_to_u64(*p)) {
                result.push(byte);
            }

        }

        for d in self.dists.iter() {

            for byte in u64_to_u8s(f64_to_u64(*d)) {
                result.push(byte);
            }

        }

        for byte in u64_to_u8s(f64_to_u64(self.mse)) {
            result.push(byte);
        }

        for byte in u64_to_u8s(self.epoch as u64) {
            result.push(byte);
        }

        result
    }

    pub fn deserialize(vec: &[u8]) -> Result<Self, ()> {

        if vec.len() != 144 {
            Err(())
        }

        else {
            let params = [
                u64_to_f64(u8s_to_u64(&vec[0..8])),
                u64_to_f64(u8s_to_u64(&vec[8..16])),
                u64_to_f64(u8s_to_u64(&vec[16..24])),
                u64_to_f64(u8s_to_u64(&vec[24..32])),
                u64_to_f64(u8s_to_u64(&vec[32..40])),
                u64_to_f64(u8s_to_u64(&vec[40..48])),
                u64_to_f64(u8s_to_u64(&vec[48..56])),
                u64_to_f64(u8s_to_u64(&vec[56..64])),
            ];
            let dists = [
                u64_to_f64(u8s_to_u64(&vec[64..72])),
                u64_to_f64(u8s_to_u64(&vec[72..80])),
                u64_to_f64(u8s_to_u64(&vec[80..88])),
                u64_to_f64(u8s_to_u64(&vec[88..96])),
                u64_to_f64(u8s_to_u64(&vec[96..104])),
                u64_to_f64(u8s_to_u64(&vec[104..112])),
                u64_to_f64(u8s_to_u64(&vec[112..120])),
                u64_to_f64(u8s_to_u64(&vec[120..128])),
            ];
            let mse = u64_to_f64(u8s_to_u64(&vec[128..136]));
            let epoch = u8s_to_u64(&vec[136..144]) as usize;

            Ok(State {
                params,
                dists,
                mse,
                epoch
            })
        }

    }

}

impl Default for State {

    fn default() -> Self {
        State::new([0.0; 8], [2.0; 8])
    }

}

pub fn init_workers<T: Clone + Send + 'static>(data: Vec<(T, f64)>, func: fn([f64; 8], T) -> f64, num_workers: usize, num_params: usize) -> Vec<Channel<T>> {
    let mut result = Vec::with_capacity(num_workers);

    for _ in 0..num_workers {
        let c = init_loop();
        c.tx_from_main.send(MessageFromMain::Initialize { data: data.clone(), func, num_params }).unwrap();
        result.push(c);
    }

    result
}

pub fn iterate<T>(state: &mut State, channels: &Vec<Channel<T>>, num_workers: usize, num_params: usize, stepss: &Vec<[i32; 8]>) -> Result<(), MPSCErr> {

    for (index, steps) in stepss.iter().enumerate() {
        match channels[index % num_workers].tx_from_main.send(MessageFromMain::GetMSE { params: state.params, dists: state.dists, steps: steps.clone() }) {
            Err(_) => {
                return Err(MPSCErr::SendFailure(index % num_workers));
            }
            _ => {}
        }
    }

    let mut best_steps: [i32; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
    let mut msg_count = 0;

    while msg_count < stepss.len() {

        for i in 0..num_workers {

            match channels[i].rx_to_main.try_recv() {
                Ok(msg) => match msg {
                    MessageToMain::NewMSE { params, steps, mse } => {

                        if mse < state.mse {
                            state.mse = mse;
                            state.params = params;
                            best_steps = steps;
                        }

                        msg_count += 1;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(MPSCErr::RecvFailure(i));
                }
                _ => {}
            }

        }

    }

    state.epoch += 1;

    for i in 0..num_params {

        match best_steps[i].abs() {
            0 => { state.dists[i] *= DIST_COEFF1; }
            1 => { state.dists[i] *= DIST_COEFF2; }
      /*2*/ _ => { state.dists[i] *= DIST_COEFF3; }
        }

        if state.dists[i] == 0.0 || state.dists[i] / state.params[i].abs() < 1e-12 {
            state.dists[i] = (state.params[i] / 4.0).abs();
        }

    }

    Ok(())
}

// `stepss` is not a typo
pub fn init_stepss(num_params: usize) -> Vec<[i32; 8]> {

    if num_params > 8 {
        return vec![];
    }

    let mut result = Vec::with_capacity(5usize.pow(num_params as u32));

    let steps_a = vec![-2, -1, 0, 1, 2];
    let steps_b = if num_params == 1 { vec![0] } else { vec![-2, -1, 0, 1, 2] };
    let steps_c = if num_params < 3 { vec![0] } else { vec![-2, -1, 0, 1, 2] };
    let steps_d = if num_params < 4 { vec![0] } else { vec![-2, -1, 0, 1, 2] };
    let steps_e = if num_params < 5 { vec![0] } else { vec![-2, -1, 0, 1, 2] };
    let steps_f = if num_params < 6 { vec![0] } else { vec![-2, -1, 0, 1, 2] };
    let steps_g = if num_params < 7 { vec![0] } else { vec![-2, -1, 0, 1, 2] };
    let steps_h = if num_params < 8 { vec![0] } else { vec![-2, -1, 0, 1, 2] };

    for a in steps_a.iter() { for b in steps_b.iter() {
        for c in steps_c.iter() { for d in steps_d.iter() {
            for e in steps_e.iter() { for f in steps_f.iter() {
                for g in steps_g.iter() { for h in steps_h.iter() {
                    result.push([*a, *b, *c, *d, *e, *f, *g, *h]);
                } }
            } }
        } }
    } }

    result
}

fn u64_to_u8s(n: u64) -> Vec<u8> {
    vec![
        ((n & 0xff00000000000000) >> 56) as u8,
        ((n & 0x00ff000000000000) >> 48) as u8,
        ((n & 0x0000ff0000000000) >> 40) as u8,
        ((n & 0x000000ff00000000) >> 32) as u8,
        ((n & 0x00000000ff000000) >> 24) as u8,
        ((n & 0x0000000000ff0000) >> 16) as u8,
        ((n & 0x000000000000ff00) >>  8) as u8,
        ((n & 0x00000000000000ff) >>  0) as u8,
    ]
}

fn u8s_to_u64(v: &[u8]) -> u64 {
    ((v[0] as u64) << 56)
    | ((v[1] as u64) << 48)
    | ((v[2] as u64) << 40)
    | ((v[3] as u64) << 32)
    | ((v[4] as u64) << 24)
    | ((v[5] as u64) << 16)
    | ((v[6] as u64) <<  8)
    | ((v[7] as u64) <<  0)
}

fn u64_to_f64(n: u64) -> f64 {
    unsafe { *(&n as *const u64 as *const f64) }
}

fn f64_to_u64(n: f64) -> u64 {
    unsafe { *(&n as *const f64 as *const u64) }
}