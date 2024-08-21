// custom_op.cu
#include <torch/extension.h>

__global__ void add_one_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0f;
    }
}

// A simple test function
void add_one(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    const int size = input.numel();
    float* data = input.data_ptr<float>();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(data, size);
}

PYBIND11_MODULE(custom_op, m) {
    m.def("add_one", &add_one, "Add one to each element of the tensor");
}
