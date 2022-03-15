import pytest
from datenaugementierung import keyboard_augmentation

@pytest.fixture()
def text():
    return "Flötenlehrerin"

def test_keyboar_aug(text):
    for i in range(10):
        assert keyboard_augmentation(text) != "Flötenlehrerin"