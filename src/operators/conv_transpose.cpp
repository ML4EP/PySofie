#include "SOFIE/ROperator_ConvTranspose.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_conv_transpose(py::module_ &m)
{
   operator_class<ROperator_ConvTranspose<float>>(m, "ROperator_ConvTranspose_float")
      .def(py::init<std::string, std::vector<std::size_t>, std::size_t, std::vector<std::size_t>,
                    std::vector<std::size_t>, std::vector<std::size_t>, std::vector<std::size_t>,
                    std::vector<std::size_t>, std::string, std::string, std::string, std::string>(),
           "autopad"_a, "dilations"_a, "group"_a, "kernel_shape"_a, "output_padding"_a, "output_shape"_a, "pads"_a,
           "strides"_a, "name_x"_a, "name_w"_a, "name_b"_a, "name_y"_a);
}
