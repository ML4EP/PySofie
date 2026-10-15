#include "SOFIE/ROperator_Identity.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_identity(py::module_ &m)
{
   bind_unary<ROperator_Identity<float>>(m, "ROperator_Identity_float");
}
