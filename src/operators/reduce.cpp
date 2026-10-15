#include "SOFIE/ROperator_Reduce.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_reduce(py::module_ &m)
{
   py::enum_<EReduceOpMode>(m, "EReduceOpMode")
      .value("ReduceMean", ReduceMean)
      .value("ReduceSum", ReduceSum)
      .value("ReduceSumSquare", ReduceSumSquare)
      .value("ReduceProd", ReduceProd)
      .value("ReduceL2", ReduceL2)
      .value("ReduceMax", ReduceMax)
      .value("ReduceMin", ReduceMin);

   auto bind_reduce = [&m](const char *name, auto tag) {
      constexpr EReduceOpMode op = decltype(tag)::value;
      operator_class<ROperator_Reduce<op>>(m, name)
         .def(py::init<int, std::vector<int64_t>, std::string, std::string, std::string>(), "keepdims"_a, "axes"_a,
              "name_x"_a, "name_axes"_a, "name_y"_a);
   };
#define SOFIE_REDUCE(K) bind_reduce("ROperator_" #K, std::integral_constant<EReduceOpMode, K>{});
   SOFIE_REDUCE(ReduceMean)
   SOFIE_REDUCE(ReduceSum)
   SOFIE_REDUCE(ReduceSumSquare)
   SOFIE_REDUCE(ReduceProd)
   SOFIE_REDUCE(ReduceL2)
   SOFIE_REDUCE(ReduceMax)
   SOFIE_REDUCE(ReduceMin)
#undef SOFIE_REDUCE
}
