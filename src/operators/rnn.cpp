#include "SOFIE/ROperator_RNN.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_rnn(py::module_ &m)
{
   operator_class<ROperator_RNN<float>>(m, "ROperator_RNN_float")
      .def(py::init<std::vector<float>, std::vector<float>, std::vector<std::string>, float, std::string, std::size_t,
                    std::size_t, std::string, std::string, std::string, std::string, std::string, std::string, std::string,
                    std::string>(),
           "activation_alpha"_a, "activation_beta"_a, "activations"_a, "clip"_a, "direction"_a, "hidden_size"_a, "layout"_a,
           "name_x"_a, "name_w"_a, "name_r"_a, "name_b"_a, "name_sequence_lens"_a, "name_initial_h"_a, "name_y"_a, "name_y_h"_a);
}
