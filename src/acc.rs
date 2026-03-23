

pub fn vec_inv(permvec: &Vec<usize>) -> Vec<usize> {
    let mut iperm = vec![0usize; permvec.len()];
    for i in 0..permvec.len() {
        iperm[permvec[i]] = i;
    }
    iperm
}