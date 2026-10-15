### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### This macro provides a simple example for the parsing of PyTorch model exported with torch.export (.pt2 file)
### into RModel object and further generating the .hxx header files for inference.
###
### \macro_code
### \macro_output
### \author Sanjiban Sengupta

import os
import torch
import torch.nn as nn

import sofie

os.makedirs("generated", exist_ok=True)

# ------------------------------------------------------------------------------
# Step 1: Create, train and save a simple PyTorch model
# ------------------------------------------------------------------------------

model = nn.Sequential(
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(2, 32)
y = torch.randn(2, 8)

for i in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

# Export the model with torch.export. The shape of the example input is the
# shape (including the batch size) of the input of the generated code.
exported = torch.export.export(model, (torch.randn(2, 32),))
torch.export.save(exported, "generated/PyTorchModel.pt2")

# ------------------------------------------------------------------------------
# Step 2: Parse the saved PyTorch .pt2 file with SOFIE
# ------------------------------------------------------------------------------

# Parsing the saved file into an RModel object. The shapes and data types of
# the input tensors are taken from the exported program.
model = sofie.pytorch.parse("generated/PyTorchModel.pt2")

# Generating inference code
model.generate()
model.output_generated("generated/PyTorchModel.hxx")

# Printing required input tensors
print("\n")
model.print_required_input_tensors()

# Printing initialized tensors (weights)
print("\n")
model.print_initialized_tensors()

# Printing intermediate tensors
print("\n")
model.print_intermediate_tensors()

# Checking if tensor already exist in model
tensor_exists = bool(model.check_if_tensor_already_exist("p_0_weight"))
print(f'\n\nTensor "p_0_weight" already exist: {str(tensor_exists).lower()}\n')

tensor_shape = model.get_tensor_shape("p_0_weight")
print('Shape of tensor "p_0_weight": ' + ",".join(str(dim) for dim in tensor_shape) + ",")

tensor_type = model.get_tensor_type("p_0_weight")
print(f'\nData type of tensor "p_0_weight": {sofie.convert_type_to_string(tensor_type)}')

# Printing generated inference code
print()
model.print_generated()
