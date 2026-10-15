#include "SOFIE/ROperator_BasicBinary.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_basic_binary(py::module_ &m)
{
   py::enum_<EBasicBinaryOperator>(m, "EBasicBinaryOperator")
      .value("Add", Add)
      .value("Sub", Sub)
      .value("Mul", Mul)
      .value("Div", Div)
      .value("Pow", Pow);

   // Element-wise binary operators, one class per operator (float only)
   auto bind_binary = [&m](const char *name, auto tag) {
      constexpr EBasicBinaryOperator op = decltype(tag)::value;
      operator_class<ROperator_BasicBinary<float, op>>(m, name)
         .def(py::init<std::string, std::string, std::string>(), "name_a"_a, "name_b"_a, "name_y"_a);
   };
   bind_binary("ROperator_BasicBinary_float_Add", std::integral_constant<EBasicBinaryOperator, Add>{});
   bind_binary("ROperator_BasicBinary_float_Sub", std::integral_constant<EBasicBinaryOperator, Sub>{});
   bind_binary("ROperator_BasicBinary_float_Mul", std::integral_constant<EBasicBinaryOperator, Mul>{});
   bind_binary("ROperator_BasicBinary_float_Div", std::integral_constant<EBasicBinaryOperator, Div>{});
   bind_binary("ROperator_BasicBinary_float_Pow", std::integral_constant<EBasicBinaryOperator, Pow>{});
}
