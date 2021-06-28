#include <torch/extension.h>

torch::Tensor hinv_setup_cuda(torch::Tensor tmp, torch::Tensor coef);
torch::Tensor hinv_mul_cuda(torch::Tensor giHig, torch::Tensor giHix);

torch::Tensor hinv_setup(torch::Tensor tmp, torch::Tensor coef) {
  return hinv_setup_cuda(tmp, coef);
}

torch::Tensor hinv_mul(torch::Tensor giHig, torch::Tensor giHix) {
  return hinv_mul_cuda(giHig, giHix);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hinv_setup", &hinv_setup, "Hinv setup (CUDA)");
  m.def("hinv_mul", &hinv_mul, "Hinv mul (CUDA)");
}
