#include <sstream>

#include "SOFIE/RModel_Base.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_types(py::module_ &m)
{
   py::enum_<ETensorType>(m, "ETensorType")
      .value("UNDEFINED", ETensorType::UNDEFINED)
      .value("FLOAT", ETensorType::FLOAT)
      .value("UINT8", ETensorType::UINT8)
      .value("INT8", ETensorType::INT8)
      .value("UINT16", ETensorType::UINT16)
      .value("INT16", ETensorType::INT16)
      .value("INT32", ETensorType::INT32)
      .value("INT64", ETensorType::INT64)
      .value("STRING", ETensorType::STRING)
      .value("BOOL", ETensorType::BOOL)
      .value("FLOAT16", ETensorType::FLOAT16)
      .value("DOUBLE", ETensorType::DOUBLE)
      .value("UINT32", ETensorType::UINT32)
      .value("UINT64", ETensorType::UINT64)
      .value("COMPLEX64", ETensorType::COMPLEX64)
      .value("COMPLEX28", ETensorType::COMPLEX28)
      .value("BFLOAT16", ETensorType::BFLOAT16);

   py::enum_<EActivationType>(m, "EActivationType")
      .value("UNDEFINED", EActivationType::UNDEFINED)
      .value("RELU", EActivationType::RELU)
      .value("SOFTMAX", EActivationType::SOFTMAX)
      .value("SIGMOID", EActivationType::SIGMOID)
      .value("LEAKYRELU", EActivationType::LEAKYRELU)
      .value("TANH", EActivationType::TANH)
      .value("ELU", EActivationType::ELU);

   py::enum_<Options>(m, "Options", py::arithmetic())
      .value("kDefault", Options::kDefault)
      .value("kNoSession", Options::kNoSession)
      .value("kNoWeightFile", Options::kNoWeightFile)
      .value("kGNN", Options::kGNN)
      .value("kGNNComponent", Options::kGNNComponent)
      .value("kProfile", Options::kProfile)
      .value("kLowRankFactorize", Options::kLowRankFactorize)
      .value("kKernelOnly", Options::kKernelOnly)
      .value("kSafetensorsWeightFile", Options::kSafetensorsWeightFile);

   py::enum_<OptimizationLevel>(m, "OptimizationLevel")
      .value("kBasic", OptimizationLevel::kBasic)
      .value("kExtended", OptimizationLevel::kExtended);

   // ------------------------------------------------------------ free functions
   m.def("convert_string_to_type", &ConvertStringToType, "type"_a);
   m.def("convert_type_to_string", &ConvertTypeToString, "type"_a);
   m.def("convert_shape_to_length", py::overload_cast<const std::vector<std::size_t> &>(&ConvertShapeToLength), "shape"_a);

   py::class_<Dim>(m, "Dim")
      .def(py::init<std::size_t>(), "dim"_a)
      .def(py::init<const std::string &, std::size_t>(), "param"_a, "dim"_a = 0)
      .def_readwrite("is_param", &Dim::isParam)
      .def_readwrite("dim", &Dim::dim)
      .def_readwrite("param", &Dim::param)
      .def("__eq__", [](const Dim &a, const Dim &b) { return a == b; })
      .def("__repr__", [](const Dim &d) {
         std::ostringstream os;
         os << d;
         return os.str();
      });
   py::implicitly_convertible<std::size_t, Dim>();
   py::implicitly_convertible<std::string, Dim>();
}
