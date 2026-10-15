#include "SOFIE/ROperator_Tanh.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_tanh(py::module_ &m)
{
   bind_unary<ROperator_Tanh<float>>(m, "ROperator_Tanh_float");
}
