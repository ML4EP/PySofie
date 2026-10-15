#include "SOFIE/ROperator_Tile.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_tile(py::module_ &m)
{
   operator_class<ROperator_Tile<float>>(m, "ROperator_Tile_float")
      .def(py::init<std::string, std::string, std::string>(), "name_repeat"_a, "name_input"_a, "name_y"_a);
}
