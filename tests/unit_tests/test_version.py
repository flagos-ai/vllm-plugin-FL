import vllm_fl.version as version


def test_public_git_exports_shape():
    assert isinstance(version.git_version, str)
    assert isinstance(version.git_info, dict)
    assert set(version.git_info.keys()) == {"id", "date"}
