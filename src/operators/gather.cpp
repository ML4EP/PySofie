#include "SOFIE/ROperator_Gather.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_gather(py::module_ &m)
{
   operator_class<ROperator_Gather>(m, "ROperator_Gather")
      .def(py::init<int64_t, std::string, std::string, std::string>(), "axis"_a, "name_x"_a, "name_indices"_a, "name_y"_a);
}
