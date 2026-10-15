"""The build backend of sofie: scikit-build-core, with a build option for the alpaka backend.
This backend understands

    pip install sofie -C alpaka=cuda      # generation of alpaka code, for nvidia
    pip install sofie -C alpaka=hip       # ... for amd

and gives it to CMake as PYSOFIE_ALPAKA_BACKEND. Everything else is left to scikit-build-core.
The environment variable SOFIE_ALPAKA_BACKEND does the same. The extras (sofie[keras], sofie[torch], sofie[full]) are the way to choose the parsers.
"""

from scikit_build_core import build as _backend

_ARCHS = ("cuda", "hip")


def _translate(config_settings):
    settings = dict(config_settings or {})
    arch = settings.pop("alpaka", None)
    if isinstance(arch, (list, tuple)):
        arch = arch[-1]
    if arch is not None:
        arch = str(arch)
        if arch not in _ARCHS:
            raise ValueError(f"-C alpaka={arch}: the backend must be one of {', '.join(_ARCHS)}")
        settings["cmake.define.PYSOFIE_ALPAKA_BACKEND"] = arch
    return settings


def get_requires_for_build_wheel(config_settings=None):
    return _backend.get_requires_for_build_wheel(_translate(config_settings))


def get_requires_for_build_editable(config_settings=None):
    return _backend.get_requires_for_build_editable(_translate(config_settings))


def get_requires_for_build_sdist(config_settings=None):
    return _backend.get_requires_for_build_sdist(_translate(config_settings))


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _backend.prepare_metadata_for_build_wheel(metadata_directory, _translate(config_settings))


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _backend.prepare_metadata_for_build_editable(metadata_directory, _translate(config_settings))


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _backend.build_wheel(wheel_directory, _translate(config_settings), metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _backend.build_editable(wheel_directory, _translate(config_settings), metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    return _backend.build_sdist(sdist_directory, _translate(config_settings))
