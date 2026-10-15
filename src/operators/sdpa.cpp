#include "SOFIE/ROperator_SDPA.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_sdpa(py::module_ &m)
{
   operator_class<ROperator_SDPA<float>>(m, "ROperator_SDPA_float")
      .def(py::init<const std::string &, const std::string &, const std::string &, const std::string &,
                    const std::string &, float, std::size_t>(),
           "name_q"_a, "name_k"_a, "name_v"_a, "name_y"_a, "name_mask"_a = "", "scale"_a = 0.0f, "num_heads"_a = 0);
}
