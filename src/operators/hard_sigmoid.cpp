#include "SOFIE/ROperator_HardSigmoid.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_hard_sigmoid(py::module_ &m)
{
   operator_class<ROperator_HardSigmoid<float>>(m, "ROperator_HardSigmoid_float")
      .def(py::init<float, float, std::string, std::string>(), "alpha"_a, "beta"_a, "name_x"_a, "name_y"_a);
}
