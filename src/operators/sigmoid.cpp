#include "SOFIE/ROperator_Sigmoid.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_sigmoid(py::module_ &m)
{
   bind_unary<ROperator_Sigmoid<float>>(m, "ROperator_Sigmoid_float");
}
