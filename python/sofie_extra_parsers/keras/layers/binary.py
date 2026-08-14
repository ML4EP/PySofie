def MakeKerasBinary(layer):
    from ROOT.TMVA.Experimental import SOFIE

    inpt = layer["layerInput"]
    if hasattr(SOFIE, "createBasicBinary"):
        return SOFIE.createBasicBinary(layer["layerDType"], layer["layerType"], inpt[0], inpt[1], layer["layerOutput"][0])

    # ROOT versions up to 6.40 don't have the createBasicBinary helper, and the
    # binary operator has to be created from the templated class on the Python side
    fLayerDType = layer["layerDType"]
    fLayerType = layer["layerType"]
    if SOFIE.ConvertStringToType(fLayerDType) != SOFIE.ETensorType.FLOAT:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator BasicBinary does not yet support input type " + fLayerDType
        )
    op_map = {
        "Add": SOFIE.EBasicBinaryOperator.Add,
        "Subtract": SOFIE.EBasicBinaryOperator.Sub,
        "Multiply": SOFIE.EBasicBinaryOperator.Mul,
    }
    if fLayerType not in op_map:
        raise RuntimeError("TMVA::SOFIE - parsing keras binary layer " + fLayerType + " is not supported")
    return SOFIE.ROperator_BasicBinary(float, op_map[fLayerType])(inpt[0], inpt[1], layer["layerOutput"][0])
