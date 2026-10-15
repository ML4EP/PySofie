#include "SOFIE/ROperator_Split.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_split(py::module_ &m)
{
   operator_class<ROperator_Split>(m, "ROperator_Split")
      .def(py::init<const std::string &, const std::string &, int, const std::vector<std::string> &>(), "name_x"_a,
           "name_split"_a, "axis"_a, "names_y"_a);
}
