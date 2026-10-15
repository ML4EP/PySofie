#include "SOFIE/ROperator_BatchNormalization.hxx"

#include "bindings.hpp"

using namespace SOFIE;

void sofie_bindings::bind_batch_normalization(py::module_ &m)
{
   operator_class<ROperator_BatchNormalization<float>>(m, "ROperator_BatchNormalization_float")
      .def(py::init<float, float, std::size_t, std::string, std::string, std::string, std::string, std::string, std::string,
                    EActivationType>(),
           "epsilon"_a, "momentum"_a, "training_mode"_a, "name_x"_a, "name_scale"_a, "name_b"_a, "name_mean"_a, "name_var"_a,
           "name_y"_a, "activation"_a = EActivationType::UNDEFINED);
}
