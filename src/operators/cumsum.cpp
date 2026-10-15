#include "SOFIE/ROperator_CumSum.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_cumsum(py::module_ &m)
{
   operator_class<ROperator_CumSum<float>>(m, "ROperator_CumSum_float")
      .def(py::init<int, int, int, const std::string &, const std::string &>(), "axis"_a, "exclusive"_a, "reverse"_a,
           "name_x"_a, "name_y"_a);
}
