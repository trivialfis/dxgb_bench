from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dxgb_bench")
    print("version:", __version__)
except PackageNotFoundError:
    # package is not installed
    pass
