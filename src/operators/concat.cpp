#include "SOFIE/ROperator_Concat.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_concat(py::module_ &m)
{
   operator_class<ROperator_Concat>(m, "ROperator_Concat")
      .def(py::init<std::vector<std::string>, int, int, std::string>(), "inputs"_a, "axis"_a, "new_axis"_a, "output"_a);
}
