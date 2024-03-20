# culina
CUDA accelerated linear algebra / CPU acceleration algorithms.
---

### CUDA
- [x] efficient tiling based large matrix multiplication
- [x] warp reduce based Matrix Vector multiplication
- [x] warp reduce based vector dot product
- [x] warp reduce
- [x] Flash attention module 1: fused QKV attention (tiling based)
    - [x] softmax(Q^T K / scale) can be easily fused
    - [ ] the extra V... well, it's a pain in the ass, TBH
- [x] coalsescing memory access benchmarking
### CPU
- [x] thread pool (condition variable and simple multi-threading)
- [x] double buffer (`std::timed_mutex` and simple multi-threading) with simple benchmarking
- [x] cache update algorithms: 
    - [x] LRU (least recently used)
    - [x] LFU (least frequently used)