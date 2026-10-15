#include "SOFIE/ROperator_InstanceNormalization.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_instance_normalization(py::module_ &m)
{
   operator_class<ROperator_InstanceNormalization<float>>(m, "ROperator_InstanceNormalization_float")
      .def(py::init<float, std::string, std::string, std::string, std::string>(), "epsilon"_a, "name_input"_a,
           "name_scale"_a, "name_bias"_a, "name_output"_a);
}
