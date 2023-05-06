#[derive(Debug)]
pub enum MPSCErr {
    SendFailure (usize),
    RecvFailure (usize),
}