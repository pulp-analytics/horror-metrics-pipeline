# Contributing

Thanks for your interest in contributing.

## How to contribute

1. Fork the repo and create your branch from `main`.
2. Make your changes, with a clear commit message.
3. Open a pull request describing what changed and why.
   CI runs `pytest -m "not slow"` on every push and PR (`make test-fast`).
4. Link any related issue.

## Installing

```bash
pip install -e ".[cpu]"                 # laptop pipeline (01-17, 19-21, 25)
pip install -e ".[cpu,tf-saliency]"     # plus 18 (TensorFlow / MSI-Net)
pip install -e ".[bedrock]"             # Nova QA (22/23/24) and S3 poster cache
pip install -e ".[all]"                 # everything
```

CI (`make test-fast`) uses the extra-free default: `pip install -e .`

## Reporting issues

Use the Issues tab. Include steps to reproduce, expected vs. actual behavior,
and relevant environment details (OS, Python version, etc.) when applicable.

## Code style

Keep changes focused and readable. Match the existing style in the file
you're editing rather than introducing a new one.

## License

By contributing, you agree that your contributions will be licensed under
the project's MIT License.
