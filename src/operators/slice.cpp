#include "SOFIE/ROperator_Slice.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_slice(py::module_ &m)
{
   // names = [starts, ends, axes, steps] (INT64 tensors; axes and steps are optional)
   operator_class<ROperator_Slice<int64_t>>(m, "ROperator_Slice_int64")
      .def(py::init<std::string, std::vector<std::string>, std::string>(), "name_data"_a, "names"_a, "name_output"_a);
}
