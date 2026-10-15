#include "SOFIE/ROperator_Gelu.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_gelu(py::module_ &m)
{
   operator_class<ROperator_Gelu>(m, "ROperator_Gelu")
      .def(py::init<std::string, std::string, std::string>(), "name_x"_a, "name_y"_a, "approximate"_a = "none");
}
