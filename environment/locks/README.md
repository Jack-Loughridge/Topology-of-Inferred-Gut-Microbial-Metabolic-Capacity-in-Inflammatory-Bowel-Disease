# Resolved modern-analysis environment lock

`requirements-linux-x86_64-py310.lock` records every installed Python package
and exact version from the clean reconstruction performed on 3 September 2026.
It was validated on Linux x86-64 with Python 3.10.12 at repository commit
`06e9b9db738552253e9645d8be9e023d4d96f578`.

Install CPU-only PyTorch from its dedicated wheel index before applying the
lock. The complete ordered commands are in `environment/README.md`.

The file pins resolved versions but does not include wheel hashes. It should
not be treated as a cross-platform lock or as proof that newly downloaded wheel
bytes are identical to those used in the audit. Its SHA-256 is:

```text
6938bab7cffdea4e0e9cd0cbaaf083319001064cf22c4994a82d0e35dcc40bf5
```
