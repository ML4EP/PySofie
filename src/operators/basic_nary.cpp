#include "SOFIE/ROperator_BasicNary.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_basic_nary(py::module_ &m)
{
   auto bind_nary = [&m](const char *name, auto tag) {
      constexpr EBasicNaryOperator op = decltype(tag)::value;
      operator_class<ROperator_BasicNary<float, op>>(m, name)
         .def(py::init<const std::vector<std::string> &, const std::string &>(), "inputs"_a, "name_y"_a);
   };
   bind_nary("ROperator_BasicNary_float_Max", std::integral_constant<EBasicNaryOperator, EBasicNaryOperator::Max>{});
   bind_nary("ROperator_BasicNary_float_Min", std::integral_constant<EBasicNaryOperator, EBasicNaryOperator::Min>{});
}
