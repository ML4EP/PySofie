// The Python module _sofie.
#include "bindings.hpp"

PYBIND11_MODULE(_sofie, m)
{
   m.doc() = "Python bindings for the standalone SOFIE inference code generator";

   // the order matters: a type has to be bound before the signatures that use it
   using namespace sofie_bindings;
   bind_types(m);
   bind_roperator(m);
   bind_rmodel(m);
   bind_rmodelparser_onnx(m);
   bind_gemm(m);
   bind_conv(m);
   bind_conv_transpose(m);
   bind_relu(m);
   bind_sigmoid(m);
   bind_tanh(m);
   bind_identity(m);
   bind_swish(m);
   bind_selu(m);
   bind_leaky_relu(m);
   bind_elu(m);
   bind_transpose(m);
   bind_reshape(m);
   bind_concat(m);
   bind_softmax(m);
   bind_pool(m);
   bind_batch_normalization(m);
   bind_layer_normalization(m);
   bind_rnn(m);
   bind_gru(m);
   bind_lstm(m);
   bind_erf(m);
   bind_hard_swish(m);
   bind_not(m);
   bind_gelu(m);
   bind_hard_sigmoid(m);
   bind_where(m);
   bind_clip(m);
   bind_cast(m);
   bind_expand(m);
   bind_gather(m);
   bind_split(m);
   bind_slice(m);
   bind_group_norm(m);
   bind_sdpa(m);
   bind_tile(m);
   bind_pad(m);
   bind_cumsum(m);
   bind_instance_normalization(m);
   bind_basic_unary(m);
   bind_basic_nary(m);
   bind_reduce(m);
   bind_comparison(m);
   bind_basic_binary(m);
}
