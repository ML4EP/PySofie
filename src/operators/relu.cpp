#include "SOFIE/ROperator_Relu.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_relu(py::module_ &m)
{
   bind_unary<ROperator_Relu<float>>(m, "ROperator_Relu_float");
}
