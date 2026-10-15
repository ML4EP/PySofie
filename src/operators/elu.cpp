#include "SOFIE/ROperator_Elu.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_elu(py::module_ &m)
{
   operator_class<ROperator_Elu<float>>(m, "ROperator_Elu_float")
      .def(py::init<float, std::string, std::string>(), "alpha"_a, "name_x"_a, "name_y"_a);
}
