#include "SOFIE/ROperator_LeakyRelu.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_leaky_relu(py::module_ &m)
{
   operator_class<ROperator_LeakyRelu<float>>(m, "ROperator_LeakyRelu_float")
      .def(py::init<float, std::string, std::string>(), "alpha"_a, "name_x"_a, "name_y"_a);
}
