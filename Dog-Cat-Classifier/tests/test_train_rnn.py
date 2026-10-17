from train_rnn import predict_label


def test_predict_label_uses_correct_binary_mapping():
    cat_label, cat_confidence = predict_label(0.2)
    dog_label, dog_confidence = predict_label(0.8)

    assert cat_label == "cat"
    assert cat_confidence == 0.8
    assert dog_label == "dog"
    assert dog_confidence == 0.8
