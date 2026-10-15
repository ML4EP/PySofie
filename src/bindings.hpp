// Shared declarations of the pybind11 bindings of SOFIE. Each bind_* function binds one part of the
// API: the types, the RModel, the ROperator base class, and one function per operator, in
// operators/<name>.cpp.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "SOFIE/ROperator.hxx"

namespace py = pybind11;
using namespace pybind11::literals;

namespace sofie_bindings {

/// The pybind11 class of an operator. py::smart_holder lets RModel::AddOperator take ownership of it.
template <typename T>
using operator_class = py::class_<T, SOFIE::ROperator, py::smart_holder>;

/// Binds an operator with one input and one output tensor and no attributes
template <typename T>
void bind_unary(py::module_ &m, const char *name)
{
   operator_class<T>(m, name).def(py::init<std::string, std::string>(), "nameX"_a, "nameY"_a);
}

void bind_types(py::module_ &m);
void bind_roperator(py::module_ &m);
void bind_rmodel(py::module_ &m);
void bind_rmodelparser_onnx(py::module_ &m);
void bind_gemm(py::module_ &m);
void bind_conv(py::module_ &m);
void bind_conv_transpose(py::module_ &m);
void bind_relu(py::module_ &m);
void bind_sigmoid(py::module_ &m);
void bind_tanh(py::module_ &m);
void bind_identity(py::module_ &m);
void bind_swish(py::module_ &m);
void bind_selu(py::module_ &m);
void bind_leaky_relu(py::module_ &m);
void bind_elu(py::module_ &m);
void bind_transpose(py::module_ &m);
void bind_reshape(py::module_ &m);
void bind_concat(py::module_ &m);
void bind_softmax(py::module_ &m);
void bind_pool(py::module_ &m);
void bind_batch_normalization(py::module_ &m);
void bind_layer_normalization(py::module_ &m);
void bind_rnn(py::module_ &m);
void bind_gru(py::module_ &m);
void bind_lstm(py::module_ &m);
void bind_erf(py::module_ &m);
void bind_hard_swish(py::module_ &m);
void bind_not(py::module_ &m);
void bind_gelu(py::module_ &m);
void bind_hard_sigmoid(py::module_ &m);
void bind_where(py::module_ &m);
void bind_clip(py::module_ &m);
void bind_cast(py::module_ &m);
void bind_expand(py::module_ &m);
void bind_gather(py::module_ &m);
void bind_split(py::module_ &m);
void bind_slice(py::module_ &m);
void bind_group_norm(py::module_ &m);
void bind_sdpa(py::module_ &m);
void bind_tile(py::module_ &m);
void bind_pad(py::module_ &m);
void bind_cumsum(py::module_ &m);
void bind_instance_normalization(py::module_ &m);
void bind_basic_unary(py::module_ &m);
void bind_basic_nary(py::module_ &m);
void bind_reduce(py::module_ &m);
void bind_comparison(py::module_ &m);
void bind_basic_binary(py::module_ &m);

} // namespace sofie_bindings
