#include "SOFIE/ROperator_Selu.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_selu(py::module_ &m)
{
   operator_class<ROperator_Selu<float>>(m, "ROperator_Selu_float")
      .def(py::init<float, float, std::string, std::string>(), "alpha"_a, "gamma"_a, "name_x"_a, "name_y"_a)
      // default SELU constants, as in ONNX
      .def(py::init([](std::string nameX, std::string nameY) {
              return std::make_unique<ROperator_Selu<float>>(1.67326319217681884765625f, 1.05070102214813232421875f,
                                                             std::move(nameX), std::move(nameY));
           }),
           "name_x"_a, "name_y"_a);
}
