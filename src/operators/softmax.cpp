#include "SOFIE/ROperator_Softmax.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_softmax(py::module_ &m)
{
   operator_class<ROperator_Softmax>(m, "ROperator_Softmax")
      .def(py::init<int64_t, std::string, std::string, bool>(), "axis"_a, "name_x"_a, "name_y"_a, "log_softmax"_a = false);
}
