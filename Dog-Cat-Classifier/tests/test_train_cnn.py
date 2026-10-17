from train_CNN import build_model


def test_build_model_uses_pretrained_backbone():
    model = build_model()

    assert model.output_shape == (None, 1)
    backbone_names = {layer.name for layer in model.layers if hasattr(layer, "name")}
    assert "mobilenetv2_1.00_224" in backbone_names
