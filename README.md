# CRINGEPACK 💀
### Complex Rust INefficient Gaussian Elimination PACKage

no cap this solver is lowkey fire (it compiles)

## what is this fr fr

CRINGEPACK is a sparse direct solver that hits different (slower). built in rust btw so you KNOW its memory safe even if the numerical accuracy is giving... questionable.

designed for symmetric complex sparse systems from EM nedelec curl-curl problems. basically maxwell's equations but make it mass production ready.

## features (real)

- 🦀 rust-pilled LU decomposition 
- partial pivoting (it pivots when the vibes are off)
- METIS nested dissection preordering (the only sigma part of this code)
- sparse triangular solves with DFS reach (lowkey impressive ngl)
- complex symmetric matrix support (imaginary numbers are valid)
- pyO3 python bindings (because we still need numpy to carry)

## features (aspirational)

- being faster than SuperLU (currently 40x slower, its giving underperformer)
- symbolic factorization (on the roadmap, trust)
- multifrontal factorization (we dont even know what this means yet no cap)
- not panicking on large matrices (working on it bestie)

## benchmarks

| solver | MDoF/s | vibe check |
|--------|--------|------------|
| SuperLU | 1000 | sigma |
| CRINGEPACK w/ METIS Frontal | 250 | Pretty based |
| CRINGEPACK w/ METIS | 12 | its a start |  
| CRINGEPACK w/ MEH ordering | 5.5 | embarrassing |
| CRINGEPACK no ordering | 0.7 | actual brain rot |

MEH = Mean Entry Heuristic. we are not making this up. we sorted by average nonzero index and called it a preordering algorithm. down horrendous.

## installation
```bash
maturin develop --release
```

the --release flag is doing all the heavy lifting here. debug mode is actually unwatchable.

## usage
```python
from cringepack import CRINGEPACK

solver = CRINGEPACK()
solver.solve(row, colptr, data, b)
# pray
```

## dependencies

- rust (obviously)
- numpy (the goat)
- pyo3 (the bridge between two worlds)
- metis-sys (the only adult in the room)
- log + env_logger (for when you need to see exactly where it all went wrong)

## roadmap

1. ~~make it compile~~ ✅
2. ~~make it correct~~ ✅ (on small matrices)
3. ~~make it correct on complex matrices~~ ✅ (after printing both re AND im)
4. make it fast (we are here, struggling)
5. make it faster than SuperLU (copium)
6. supernodal factorization (delulu is the solulu)

## acknowledgments

- Tim Davis's book "Direct Methods for Sparse Linear Systems" — the only reason any of this works
- Claude — pair programmed the entire thing, never once judged us for the vec semicolon incident
- SuperLU — for being the benchmark we will one day catch (cope)
- the rust borrow checker — for teaching us about lifetimes the hard way

## license

MIT. do whatever you want with it. if you use CRINGEPACK in production you are either very brave or very lost.

## citation

if you somehow use this in a paper:
```bibtex
@software{cringepack,
  title={CRINGEPACK: Complex Rust INefficient Gaussian Elimination PACKage},
  author={a person who learned rust last week},
  year={2026},
  note={it works on my machine}
}
```

---

*"its not a bug its a ✨feature✨"* — the commit messages, probably
