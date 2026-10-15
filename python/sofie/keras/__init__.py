def get_keras_version() -> str:

    import keras

    return keras.__version__


from .parser import parse  # noqa: E402  (the parser imports get_keras_version from this package)

__all__ = ["get_keras_version", "parse"]
