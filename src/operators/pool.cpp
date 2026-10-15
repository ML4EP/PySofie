#include "SOFIE/ROperator_Pool.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_pool(py::module_ &m)
{
   py::enum_<PoolOpMode>(m, "PoolOpMode")
      .value("InvalidPool", InvalidPool)
      .value("MaxPool", MaxPool)
      .value("AveragePool", AveragePool)
      .value("GlobalAveragePool", GlobalAveragePool);

   py::class_<RAttributes_Pool>(m, "RAttributes_Pool")
      .def(py::init<>())
      .def_readwrite("auto_pad", &RAttributes_Pool::auto_pad)
      .def_readwrite("ceil_mode", &RAttributes_Pool::ceil_mode)
      .def_readwrite("count_include_pad", &RAttributes_Pool::count_include_pad)
      .def_readwrite("storage_order", &RAttributes_Pool::storage_order)
      .def_readwrite("dilations", &RAttributes_Pool::dilations)
      .def_readwrite("kernel_shape", &RAttributes_Pool::kernel_shape)
      .def_readwrite("pads", &RAttributes_Pool::pads)
      .def_readwrite("strides", &RAttributes_Pool::strides);

   operator_class<ROperator_Pool<float>>(m, "ROperator_Pool_float")
      .def(py::init<PoolOpMode, RAttributes_Pool, std::string, std::string>(), "mode"_a, "attr"_a, "name_x"_a, "name_y"_a);
}
