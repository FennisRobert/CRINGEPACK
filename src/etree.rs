use crate::sparse::CCMatrixOwned;
use crate::acc::vec_inv;

fn firstdesc(n: usize, parent: &Vec<usize>, post: &Vec<usize>, first: &mut Vec<usize>, level: &mut Vec<usize>) {
    first.fill(usize::MAX);
    for k in 0..n {
        let i = post[k];
        let mut len: usize = 0;
        let mut r = i;
        loop {
            if r==usize::MAX || first[r] != usize::MAX {
                break;
            }
            first[r] = k;
            len += 1;
            r = parent[r];
        }

        if r == usize::MAX {
            if len > 0 {
                len = len -1;
            }
        } else {
            len = len + level[r];
        }

        let mut s = i;
        while s != r {
            level[s] = len;
            if len > 0 {
                len = len - 1;
            }
            s = parent[s]
        }

    }

}


pub fn tree_depth_first_search(j: usize, k: usize, head: &mut Vec<usize>, next: &Vec<usize>, post: &mut Vec<usize>, stack: &mut Vec<usize>) -> usize {
    let mut top = 0 as i64;
    
    let mut k = k;
    stack[0] = j;
    while top >= 0 {
        let p = stack[top as usize];
        let i = head[p];
        if i == usize::MAX {
            top = top - 1;
            post[k] = p;
            k += 1usize;
        }else{
            head[p] = next[i];
            top += 1;
            stack[top as usize] = i;
            
        }
    }
    k

}


pub fn elimination_tree(matrix: &CCMatrixOwned) -> Vec<usize> {
    let n = matrix.n;
    let mut parent = vec![usize::MAX; n];
    let mut ancestor = vec![0usize; n];

    for j in 0..n {
        ancestor[j] = j;
        let i1 = matrix.indptr[j] as usize;
        let i2 = matrix.indptr[j + 1] as usize;
        for idx in i1..i2 {
            let i = matrix.rows[idx] as usize;
            if i >= j {
                continue; // only look at upper triangle
            }
            // Walk from i up to j, compressing the path
            let mut node = i;
            loop {
                let next = ancestor[node];
                if next == j {
                    break; // already linked to j
                }
                ancestor[node] = j; // path compression
                if next == node {
                    // node was a root, make j its parent
                    parent[node] = j;
                    break;
                }
                node = next;
            }
        }
    }
    parent
}


fn leaf(i: usize, j: usize, first: &Vec<usize>, maxfirst: &mut Vec<usize>, prevleaf: &mut Vec<usize>, ancestor: &mut Vec<usize>,) -> (usize, usize) {
    let q = 0usize;
    let mut jleaf = 0usize;
    if i <= j || first[j] == usize::MAX || (maxfirst[i] != usize::MAX && first[j] <= maxfirst[i]) {
        return (usize::MAX, jleaf);
    }
    maxfirst[i] = first[j];
    let jprev = prevleaf[i];
    prevleaf[i] = j;

    if jprev == usize::MAX {
        jleaf = 1;
        return (i, jleaf);
    }

    jleaf = 2;
    let mut q = jprev;
    while q != ancestor[q] {
        q = ancestor[q];
    }
    let mut s = jprev;
    while s!= q{
        let sparent = ancestor[s];
        ancestor[s] = q;
        s = sparent;
    }
    (q, jleaf)
}


pub fn postorder(etree: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let n = etree.len();
    let mut k = 0usize;
    let mut head = vec![usize::MAX; n];
    let mut next = vec![0usize; n];
    let mut stack = vec![0usize; n];
    let mut post_ordering = vec![0usize; n];
    for j in (0..(n-1)).rev() {
        if etree[j] == usize::MAX {
            continue
        }
        next[j] = head[etree[j]];
        head[etree[j]] = j;
    }
    for j in 0..n {
        if etree[j] != usize::MAX {
            continue;
        }
        k = tree_depth_first_search(j, k, &mut head, &next, &mut post_ordering, &mut stack);
    }
    let mut iperm = vec![0usize; n];
    for i in 0..n {
        iperm[post_ordering[i]] = i;
    }
    let etree_post: Vec<usize> = (0..n).map(|i| {
        let old = post_ordering[i];
        if etree[old] == usize::MAX {
            usize::MAX
        }else {
            iperm[etree[old]]
        }
    }).collect();
    (etree_post, post_ordering)
}

pub fn etree_post(matrix: &CCMatrixOwned) -> (Vec<usize>, Vec<usize>) {
    postorder(&elimination_tree(&matrix))
}


pub fn col_non_zeros(
    matrix: &CCMatrixOwned,
    etree: &Vec<usize>,
    post: &Vec<usize>,
) -> Vec<usize> {
    let n = matrix.n;

    let mut ancestor = vec![0usize; n];
    let mut maxfirst = vec![usize::MAX; n];
    let mut prevleaf = vec![usize::MAX; n];
    let mut first = vec![usize::MAX; n];
    let mut delta = vec![0i64; n];

    for k in 0..n {
        let mut j = post[k];
        if first[j] == usize::MAX {
            delta[j] = 1;
        }
        while j != usize::MAX && first[j] == usize::MAX {
            first[j] = k;
            j = etree[j];
        }
    }

    for i in 0..n {
        ancestor[i] = i;
    }

    for k in 0..n {
        let j = post[k];

        if etree[j] != usize::MAX {
            delta[etree[j]] -= 1;
        }

        let p1 = matrix.indptr[j] as usize;
        let p2 = matrix.indptr[j + 1] as usize;
        for p in p1..p2 {
            let i = matrix.rows[p] as usize;
            let (q, jleaf) = leaf(
                i, j, &first, &mut maxfirst, &mut prevleaf, &mut ancestor,
            );
            if jleaf >= 1 {
                delta[j] += 1;
            }
            if jleaf == 2 {
                delta[q] -= 1;
            }
        }

        if etree[j] != usize::MAX {
            ancestor[j] = etree[j];
        }
    }

    for j in 0..n {
        if etree[j] != usize::MAX {
            delta[etree[j]] += delta[j];
        }
    }

    delta.iter().map(|&x| x as usize).collect()
}


struct RowsFirstInCol {
    head: Vec<usize>,
    next: Vec<usize>,
}

impl RowsFirstInCol {
    fn new(mat: &CCMatrixOwned, post_ordering: &Vec<usize>) -> Self {
        let n = mat.n;
        let mut head = vec![usize::MAX; n];
        let mut next = vec![usize::MAX; n];
        let inv_post = vec_inv(&post_ordering);
        
        for icol in 0..n {
            let i1 = mat.indptr[icol] as usize;
            let i2 = mat.indptr[icol + 1] as usize;
            let mut k = usize::MAX;
            for i in i1..i2 {
                k = k.min(inv_post[mat.rows[i]]);
            }
            next[icol] = head[k];
            head[k] = icol;
        }
        RowsFirstInCol { head, next }
    }
    

    fn printself(&self) {
        for i in 0..self.head.len() {
            let mut nums: Vec<usize> = Vec::with_capacity(self.head.len());
            let mut cur = self.head[i];

            while cur != usize::MAX {
                nums.push(cur + 1usize);
                cur = self.next[cur];
            }
            println!("Col {} = {:?}", i, nums);
        }
        println!("HEAD: {:?}", self.head);
        println!("NEXT: {:?}", self.next);
        
    }
}

pub fn row_non_zeros(matrix: &CCMatrixOwned, etree: &Vec<usize>, post: &Vec<usize>) -> Vec<usize> {
    
    let n = matrix.n;
    let ap = &matrix.indptr;
    let ai = &matrix.rows;
    let mut ancestors = vec![0usize; n];
    let mut maxfirst = vec![0usize; n];
    let mut prevleaf = vec![0usize; n];
    let mut first = vec![0usize; n];
    let mut level = vec![0usize; n];
    let mut n_in_row = vec![0usize; n];
    firstdesc(n, etree, &post, &mut first, &mut level);
    for i in 0..n {
        n_in_row[i] = 1;
        prevleaf[i] = usize::MAX;
        maxfirst[i] = usize::MAX;
        ancestors[i] = i;
    }

    for k in 0..n {
        let j = post[k];
        for p in ap[j] .. ap[j+1] {
            let i = ai[p];
            println!("Loop k={}", k);
            let (q, jleaf) = leaf(i, j, &mut first, &mut maxfirst, &mut prevleaf, &mut ancestors);
            if jleaf != 0usize {
                println!("levels = {}, {}", level[j], level[q]);
                n_in_row[i] += level[j] - level[q];
            }
        }
        if etree[j] != usize::MAX {
            ancestors[j] = etree[j];
        }
    }

    n_in_row
}
