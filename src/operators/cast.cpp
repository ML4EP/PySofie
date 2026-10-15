#include "SOFIE/ROperator_Cast.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_cast(py::module_ &m)
{
   operator_class<ROperator_Cast>(m, "ROperator_Cast")
      .def(py::init<ETensorType, std::string, std::string>(), "type"_a, "name_x"_a, "name_y"_a);
}
