"""Heavy runtime adapters for the NLF worker (torch, PyAV, OpenCV, pyrender, boto3, supabase),
each implementing one protocol from ``src/nlf_worker/interfaces.py``. Run under ``.venv-nlf``.

Import the submodule you need directly; this ``__init__`` stays empty so importing one adapter
never drags in the others' dependencies. Nothing outside ``runtime/`` imports this package.
"""
