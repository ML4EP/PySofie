#include "SOFIE/ROperator_GroupNorm.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_group_norm(py::module_ &m)
{
   operator_class<ROperator_GroupNorm<float>>(m, "ROperator_GroupNorm_float")
      .def(py::init<int, float, const std::string &, const std::string &, const std::string &, const std::string &>(),
           "num_groups"_a, "epsilon"_a, "name_x"_a, "name_scale"_a, "name_bias"_a, "name_y"_a);
}
