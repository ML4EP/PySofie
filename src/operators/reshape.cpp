#include "SOFIE/ROperator_Reshape.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_reshape(py::module_ &m)
{
   py::enum_<ReshapeOpMode>(m, "ReshapeOpMode")
      .value("Reshape", Reshape)
      .value("Flatten", Flatten)
      .value("Squeeze", Squeeze)
      .value("Unsqueeze", Unsqueeze);

   operator_class<ROperator_Reshape>(m, "ROperator_Reshape")
      .def(py::init<ReshapeOpMode, int, std::string, std::string, std::string>(), "op_mode"_a, "attr_value"_a, "name_data"_a,
           "name_input2"_a, "name_output"_a)
      .def(py::init<ReshapeOpMode, std::vector<int64_t>, std::string, std::string>(), "op_mode"_a, "attr_axes"_a, "name_data"_a,
           "name_output"_a);
}
