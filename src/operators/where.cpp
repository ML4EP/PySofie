#include "SOFIE/ROperator_Where.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_where(py::module_ &m)
{
   operator_class<ROperator_Where<float>>(m, "ROperator_Where_float")
      .def(py::init<std::string, std::string, std::string, std::string>(), "name_c"_a, "name_x"_a, "name_y"_a, "name_z"_a);
}
