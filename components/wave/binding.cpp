#include <torch/extension.h>

torch::Tensor wave_hadamard(torch::Tensor input, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("hadamard_transform", &wave_hadamard,
               pybind11::arg("x"), pybind11::arg("scale") = 1.0f,
               "Wave Hadamard transform for FP16/BF16 widths 32/64/128/256");
}
