"""Import-time placeholder; TRELLIS.2 MPS generation does not use rendering."""


def __getattr__(name):
    raise RuntimeError(
        f"nvdiffrast.{name} is unavailable on MPS; use mesh export instead of the CUDA renderer"
    )
