#include "SOFIE/ROperator_Transpose.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_transpose(py::module_ &m)
{
   operator_class<ROperator_Transpose>(m, "ROperator_Transpose")
      .def(py::init<std::vector<int64_t>, std::string, std::string>(), "perm"_a, "name_data"_a, "name_output"_a);
}
