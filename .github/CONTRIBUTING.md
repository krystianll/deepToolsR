# Contributing to deepToolsR

Bug reports and pull requests are welcome.

## Reporting a problem

Open an issue with the deepToolsR version (for example `bamCoverageR --version`),
your Python version and operating system, the full command, and the complete
output it printed. A small input file that reproduces the problem helps most.

## Changing the code

* Fork the repository and create a branch from `main`.
* Build and install it in a virtual environment: `python -m pip install -e .`
* Add or update tests for your change and run them with `python -m pytest tests`.
* Keep `flake8` clean.
* Describe user-visible changes in `CHANGES.txt`.
* Open a pull request that explains what changed and why.
