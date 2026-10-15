#include "SOFIE/ROperator_Expand.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_expand(py::module_ &m)
{
   operator_class<ROperator_Expand<float>>(m, "ROperator_Expand_float")
      .def(py::init<std::string, std::string, std::string>(), "name_x"_a, "name_shape"_a, "name_y"_a);
}
