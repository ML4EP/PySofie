#include "SOFIE/ROperator_Erf.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_erf(py::module_ &m)
{
   bind_unary<ROperator_Erf<float>>(m, "ROperator_Erf_float");
}
