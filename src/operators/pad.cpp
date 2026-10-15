#include "SOFIE/ROperator_Pad.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_pad(py::module_ &m)
{
   // pads: INT64 tensor [x1_begin, x2_begin, ..., x1_end, x2_end, ...], constantValue: float tensor of one element
   operator_class<ROperator_Pad<float>>(m, "ROperator_Pad_float")
      .def(py::init<const std::string &, const std::string &, const std::string &, const std::string &,
                    const std::string &, const std::string &>(),
           "name_x"_a, "name_pads"_a, "name_constant_value"_a, "name_axes"_a, "name_y"_a, "mode"_a = "constant");
}
