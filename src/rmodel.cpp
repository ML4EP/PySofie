#include <pybind11/iostream.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "SOFIE/RModel.hxx"

#include "bindings.hpp"

using namespace SOFIE;

namespace {

const char *NumpyDType(ETensorType type)
{
   switch (type) {
   case ETensorType::FLOAT: return "float32";
   case ETensorType::DOUBLE: return "float64";
   case ETensorType::UINT8: return "uint8";
   case ETensorType::INT8: return "int8";
   case ETensorType::UINT16: return "uint16";
   case ETensorType::INT16: return "int16";
   case ETensorType::INT32: return "int32";
   case ETensorType::INT64: return "int64";
   case ETensorType::UINT32: return "uint32";
   case ETensorType::UINT64: return "uint64";
   case ETensorType::BOOL: return "bool";
   default: throw std::invalid_argument("SOFIE: unsupported tensor type for initialized tensor: " + ConvertTypeToString(type));
   }
}

// Copy the contents of any array-like object into a buffer of the given tensor type. numpy converts
// the data (any array, tensor-like or sequence; any dtype and byte order) to a contiguous array of the
// type of the tensor. numpy is imported here, so only code that adds tensor data needs it.
std::shared_ptr<void> MakeBuffer(const std::string &name, ETensorType type, const std::vector<std::size_t> &shape,
                                 const py::object &data)
{
   py::object np;
   try {
      np = py::module_::import("numpy");
   } catch (py::error_already_set &e) {
      if (!e.matches(PyExc_ImportError))
         throw;
      throw std::runtime_error("SOFIE: adding the data of tensor " + name + " needs numpy, which is not installed");
   }
   py::object arr = np.attr("ascontiguousarray")(data, "dtype"_a = NumpyDType(type));
   py::buffer_info info = py::buffer(arr).request();
   const std::size_t length = ConvertShapeToLength(shape);
   if (static_cast<std::size_t>(info.size) != length)
      throw std::invalid_argument("SOFIE: tensor " + name + " has " + std::to_string(info.size) +
                                  " elements, but its shape needs " + std::to_string(length));
   if (static_cast<std::size_t>(info.itemsize) != GetTypeSize(type))
      throw std::logic_error("SOFIE: unexpected element size of the data of tensor " + name);
   const std::size_t nbytes = length * GetTypeSize(type);
   std::shared_ptr<void> buffer(std::malloc(nbytes ? nbytes : 1), std::free);
   std::memcpy(buffer.get(), info.ptr, nbytes);
   return buffer;
}

void AddInitializedTensor(RModel &model, const std::string &name, ETensorType type, const std::vector<std::size_t> &shape,
                          const py::object &data)
{
   model.AddInitializedTensor(name, type, shape, MakeBuffer(name, type, shape, data));
}

void AddConstantTensor(RModel &model, const std::string &name, ETensorType type, const std::vector<std::size_t> &shape,
                       const py::object &data)
{
   model.AddConstantTensor(name, type, shape, MakeBuffer(name, type, shape, data));
}

void Capture(RModel &model, void (RModel::*print)() const)
{
   // the Print* methods write to std::cout, which is not Python's sys.stdout
   py::scoped_ostream_redirect redirect(std::cout, py::module_::import("sys").attr("stdout"));
   (model.*print)();
}

} // namespace

void sofie_bindings::bind_rmodel(py::module_ &m)
{
   py::class_<RModel>(m, "RModel")
      .def(py::init<std::string, std::string>(), "name"_a, "parsetime"_a)
      .def("add_input_tensor_info", py::overload_cast<std::string, ETensorType, std::vector<std::size_t>>(&RModel::AddInputTensorInfo),
           "name"_a, "type"_a, "shape"_a)
      .def("add_input_tensor_info", py::overload_cast<std::string, ETensorType, std::vector<Dim>>(&RModel::AddInputTensorInfo),
           "name"_a, "type"_a, "shape"_a)
      .def("add_input_tensor_name", &RModel::AddInputTensorName, "name"_a)
      .def("add_output_tensor_name_list", &RModel::AddOutputTensorNameList, "names"_a)
      // Takes ownership of the operator (hence py::smart_holder): the Python object is left empty, as
      // with a C++ std::move into the std::unique_ptr parameter.
      .def("add_operator", &RModel::AddOperator, "op"_a, "order_execution"_a = -1)
      .def("add_initialized_tensor", &AddInitializedTensor, "name"_a, "type"_a, "shape"_a, "data"_a)
      .def("add_constant_tensor", &AddConstantTensor, "name"_a, "type"_a, "shape"_a, "data"_a)
      .def("set_not_writable_initialized_tensor", &RModel::SetNotWritableInitializedTensor, "name"_a)
      .def("mark_scalar_tensor", &RModel::MarkScalarTensor, "name"_a)
      .def("add_blas_routines",
           [](RModel &self, const py::iterable &routines) {
              std::vector<std::string> names;
              for (auto r : routines)
                 names.push_back(r.cast<std::string>());
              self.AddBlasRoutines(names);
           },
           "routines"_a)
      .def("add_needed_std_lib", &RModel::AddNeededStdLib, "libname"_a)
      .def("check_if_tensor_already_exist", &RModel::CheckIfTensorAlreadyExist, "name"_a)
      .def("is_initialized_tensor", &RModel::IsInitializedTensor, "name"_a)
      .def("get_tensor_shape", &RModel::GetTensorShape, "name"_a)
      .def("get_tensor_type", &RModel::GetTensorType, "name"_a)
      .def("get_input_tensor_names", &RModel::GetInputTensorNames)
      .def("get_output_tensor_names", &RModel::GetOutputTensorNames)
      .def("get_name", &RModel::GetName)
      .def("initialize", py::overload_cast<int, bool>(&RModel::Initialize), "batch_size"_a = -1, "verbose"_a = false)
      .def(
         "generate",
         [](RModel &self, Options options, int batch_size, bool verbose) { self.Generate(options, batch_size, verbose); },
         "options"_a = Options::kDefault, "batch_size"_a = -1, "verbose"_a = false)
      // Generates the heterogeneous inference code that uses alpaka; get it with output_generated/return_generated.
      // Opt-in at build time (PYSOFIE_ENABLE_ALPAKA): otherwise the method raises an error that says so.
      .def(
         "generate_gpu_alpaka",
         [](RModel &self, Options options, int batch_size, bool verbose) {
#ifdef PYSOFIE_ENABLE_ALPAKA
            self.GenerateGPU_ALPAKA(options, batch_size, verbose);
#else
            (void)self, (void)options, (void)batch_size, (void)verbose;
            throw std::runtime_error("sofie was built without the generation of alpaka code: reinstall it for a backend, "
                                     "pip install sofie -C alpaka=cuda (cuda or hip; "
                                     "or ./install.sh prod|dev --alpaka --arch cuda|hip)");
#endif
         },
         "options"_a = Options::kDefault, "batch_size"_a = -1, "verbose"_a = false)
      .def("output_generated", [](RModel &self, const std::string &filename, bool append) { self.OutputGenerated(filename, append); },
           "filename"_a = "", "append"_a = false)
      .def("return_generated", &RModel::ReturnGenerated)
      .def("print_generated", [](RModel &self) { py::print(self.ReturnGenerated(), "end"_a = ""); })
      .def("print_required_input_tensors", [](RModel &self) { Capture(self, &RModel::PrintRequiredInputTensors); })
      .def("print_initialized_tensors", [](RModel &self) { Capture(self, &RModel::PrintInitializedTensors); })
      .def("print_intermediate_tensors", [](RModel &self) { Capture(self, &RModel::PrintIntermediateTensors); })
      .def("print_dynamic_tensors", [](RModel &self) { Capture(self, &RModel::PrintDynamicTensors); })
      .def("print_output_tensors", [](RModel &self) { Capture(self, &RModel::PrintOutputTensors); })
      .def("print_summary", [](RModel &self) { Capture(self, &RModel::PrintSummary); });

   // whether this build can generate alpaka code (sofie.with_alpaka), and the backend it is for (sofie.alpaka_backend)
#ifdef PYSOFIE_ENABLE_ALPAKA
   m.attr("with_alpaka") = true;
   m.attr("alpaka_backend") = PYSOFIE_ALPAKA_BACKEND;
#else
   m.attr("with_alpaka") = false;
   m.attr("alpaka_backend") = py::none();
#endif
}
