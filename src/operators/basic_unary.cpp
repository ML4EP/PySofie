#include "SOFIE/ROperator_BasicUnary.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_basic_unary(py::module_ &m)
{
   auto bind_kind = [&m](const char *name, auto tag) {
      constexpr EBasicUnaryOperator op = decltype(tag)::value;
      operator_class<ROperator_BasicUnary<float, op>>(m, name)
         .def(py::init<std::string, std::string>(), "name_x"_a, "name_y"_a);
   };
#define SOFIE_UNARY(K) \
   bind_kind("ROperator_BasicUnary_float_" #K, std::integral_constant<EBasicUnaryOperator, EBasicUnaryOperator::k##K>{});
   SOFIE_UNARY(Reciprocal)
   SOFIE_UNARY(Sqrt)
   SOFIE_UNARY(Neg)
   SOFIE_UNARY(Exp)
   SOFIE_UNARY(Log)
   SOFIE_UNARY(Sin)
   SOFIE_UNARY(Cos)
   SOFIE_UNARY(Abs)
   SOFIE_UNARY(Floor)
#undef SOFIE_UNARY
}
