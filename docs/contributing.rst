Contributing to MLArena
======================

We welcome contributions to MLArena! This document provides guidelines and instructions for contributing.

Getting Started
--------------

1. Fork the repository
2. Clone your fork:
   .. code-block:: bash
      git clone https://github.com/yourusername/mlarena.git
3. Create a new branch:
   .. code-block:: bash
      git checkout -b feature/your-feature-name
4. Set up your development environment:
   .. code-block:: bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

Development Guidelines
-------------------

Code Style
~~~~~~~~~

We follow PEP 8 guidelines and use the following tools:

* `black` for code formatting
* `flake8` for linting
* `isort` for import sorting
* `mypy` for type checking

Before submitting a pull request, run:

.. code-block:: bash

   black .
   isort .
   flake8
   mypy .

Testing
~~~~~~

We use `pytest` for testing. Write tests for new features and ensure all tests pass:

.. code-block:: bash

   pytest

Documentation
~~~~~~~~~~~

* Update the relevant documentation files in the `docs` directory
* Add docstrings to all new functions and classes
* Include examples in the documentation
* Update the changelog

Pull Request Process
------------------

1. Update the README.md with details of changes if needed
2. Update the documentation with any new features
3. Add tests for new functionality
4. Ensure the test suite passes
5. Update the CHANGELOG.md
6. Submit a pull request

Code Review
----------

* All pull requests require at least one review
* Address review comments promptly
* Keep pull requests focused and small
* Write clear commit messages

Release Process
-------------

1. Update version in setup.py
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Build and upload to PyPI:
   .. code-block:: bash
      python -m build
      twine upload dist/*

Project Structure
--------------

::

   mlarena/
   ├── .github/           # GitHub Actions and workflows
   ├── docs/              # Documentation
   │   ├── api.rst
   │   ├── conf.py
   │   ├── contributing.rst
   │   ├── MIGRATION_GUIDE.md
   │   └── images/
   ├── examples/          # Example notebooks and scripts
   ├── mlarena/          # Main package
   │   ├── __init__.py
   │   ├── preprocessor.py
   │   ├── pipeline.py
   │   └── utils/        # Utility modules
   │       ├── __init__.py
   │       ├── data_utils.py     # Data manipulation utilities
   │       ├── io_utils.py       # Input/Output utilities
   │       └── plot_utils.py     # Plotting utilities
   ├── tests/            # Test files
   │   ├── test_pipeline.py
   │   ├── test_preprocessor.py
   │   ├── test_data_utils.py
   │   ├── test_io_utils.py
   │   └── test_plot_utils.py
   ├── pyproject.toml    # Package configuration and dependencies
   ├── poetry.lock       # Lock file for reproducible builds
   ├── README.md         # Project README
   ├── CHANGELOG.md      # Version history
   ├── LICENSE          # License file
   ├── .gitignore       # Git ignore rules
   ├── .flake8         # Flake8 configuration
   ├── mypy.ini        # MyPy configuration
   └── pytest.ini      # Pytest configuration

Contact
-------

If you have any questions or need help, please:

1. Open an issue
2. Join our community chat
3. Contact the maintainers

Thank you for contributing to MLArena! 