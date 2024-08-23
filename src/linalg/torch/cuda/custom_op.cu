// custom_op.cu
#include <iostream>
#include <torch/extension.h>

torch::Tensor binary_op(torch::Tensor input1, torch::Tensor input2) {
    TORCH_CHECK(input1.is_cuda(),       "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input2.is_contiguous(), "Input tensor must be contiguous");

    const int size = input1.numel();
    std::cout << "Tensor 1 with shape " << input1.sizes() << std::endl;
    std::cout << "Tensor 2 with shape " << input2.sizes() << std::endl;
    float* data1 = input1.data_ptr<float>();
    float* data2 = input2.data_ptr<float>();
    return input1;
}

torch::Tensor unary_op(torch::Tensor input1) {
    TORCH_CHECK(input1.is_cuda(),       "Input tensor must be a CUDA tensor");

    const int size = input1.numel();
    std::cout << "Tensor 1 with shape " << input1.sizes() << std::endl;
    float* data1 = input1.data_ptr<float>();
    return input1;
}

PYBIND11_MODULE(custom_op, m) {
    m.def("binary_op", &binary_op, "Binary operation.");
    m.def("unary_op", &unary_op, "Unary operation.");
}
