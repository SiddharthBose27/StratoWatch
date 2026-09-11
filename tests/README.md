# Phase 1 Verification Tests

These tests are read-only contract checks for the frozen StratoWatch baseline. They do not train models, regenerate datasets, rewrite checkpoints, or modify research outputs.

Run them with the Python standard library:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The artifact and model tests inspect the runtime files currently present in the workspace. A failure is a baseline finding unless the test itself is changed in a later phase.
