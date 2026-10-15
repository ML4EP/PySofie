#include <pybind11/iostream.h>

#include "SOFIE/RModelParser_ONNX.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_rmodelparser_onnx(py::module_ &m)
{
   // SOFIE's own parser of ONNX models (.onnx files).
   py::class_<RModelParser_ONNX>(m, "RModelParser_ONNX")
      .def(py::init<>())
      .def("parse", py::overload_cast<const std::string &, bool>(&RModelParser_ONNX::Parse), "filename"_a,
           "verbose"_a = false)
      .def(
         "check_model",
         [](RModelParser_ONNX &self, const std::string &filename, bool verbose) {
            // Redirect the output to Python's stdout
            py::scoped_ostream_redirect redirect(std::cout, py::module_::import("sys").attr("stdout"));
            return self.CheckModel(filename, verbose);
         },
         "filename"_a, "verbose"_a = false)
      .def("set_external_data_file", &RModelParser_ONNX::SetExternalDataFile, "filename"_a);
}
