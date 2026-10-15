#include "SOFIE/ROperator_Clip.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_clip(py::module_ &m)
{
   operator_class<ROperator_Clip<float>>(m, "ROperator_Clip_float")
      // bounds known at parse time
      .def(py::init<std::string, std::string, float, float>(), "name_x"_a, "name_y"_a, "min"_a, "max"_a);
}
