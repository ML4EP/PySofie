#include "SOFIE/ROperator_LayerNormalization.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_layer_normalization(py::module_ &m)
{
   operator_class<ROperator_LayerNormalization<float>>(m, "ROperator_LayerNormalization_float")
      .def(py::init<int, float, std::size_t, const std::string &, const std::string &, const std::string &,
                    const std::string &, const std::string &, const std::string &>(),
           "axis"_a, "epsilon"_a, "stash_type"_a, "name_x"_a, "name_scale"_a, "name_b"_a, "name_y"_a, "name_mean"_a,
           "name_inv_std_dev"_a);
}
