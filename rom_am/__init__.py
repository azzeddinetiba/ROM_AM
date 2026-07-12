from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rom_am")
except PackageNotFoundError:
    # package is not installed
    pass
