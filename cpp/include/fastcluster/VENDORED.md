# fastcluster Ward core

`fastcluster.cpp` is based on fastcluster v1.3.0,
commit `e8eccdc8bc8ec0558cf0c672220c6f50bbce51c8`:
<https://github.com/fastcluster/fastcluster/tree/v1.3.0>.
`COPYING.txt` is the upstream two-clause BSD license. The vendored source has
no local modifications. `cpp/cluster.cpp` includes its stored-distance
`NN_chain_core` for Ward and implements the Python wrapper's distance
squaring, square-rooting, and SciPy linkage conversion outside this file.
