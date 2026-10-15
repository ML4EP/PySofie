#include "SOFIE/ROperator_Comparison.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_comparison(py::module_ &m)
{
   auto bind_cmp = [&m](const char *name, auto tag) {
      constexpr EComparisonOperator op = decltype(tag)::value;
      operator_class<ROperator_Comparison<float, op>>(m, name)
         .def(py::init<const std::string &, const std::string &, const std::string &>(), "name_x1"_a, "name_x2"_a,
              "name_y"_a);
   };
#define SOFIE_CMP(K) bind_cmp("ROperator_Comparison_float_" #K, std::integral_constant<EComparisonOperator, K>{});
   SOFIE_CMP(Eq)
   SOFIE_CMP(Less)
   SOFIE_CMP(LessEq)
   SOFIE_CMP(Greater)
   SOFIE_CMP(GreaterEq)
#undef SOFIE_CMP
}
