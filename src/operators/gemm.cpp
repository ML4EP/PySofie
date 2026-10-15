#include "SOFIE/ROperator_Gemm.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_gemm(py::module_ &m)
{
   using Gemm = ROperator_Gemm<float>;
   operator_class<Gemm>(m, "ROperator_Gemm_float")
      .def(py::init<float, float, int, int, std::string, std::string, std::string, std::string, EActivationType>(), "alpha"_a,
           "beta"_a, "trans_a"_a, "trans_b"_a, "name_a"_a, "name_b"_a, "name_c"_a, "name_y"_a,
           "activation"_a = EActivationType::UNDEFINED)
      .def(py::init<float, float, int, int, std::string, std::string, std::string, EActivationType>(), "alpha"_a, "beta"_a,
           "trans_a"_a, "trans_b"_a, "name_a"_a, "name_b"_a, "name_y"_a, "activation"_a = EActivationType::UNDEFINED);
}
