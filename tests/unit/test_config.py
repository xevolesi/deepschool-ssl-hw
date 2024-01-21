from source.utils.general import read_config


def test_read_config(get_test_config_path):
    config = read_config(get_test_config_path)
    assert config
