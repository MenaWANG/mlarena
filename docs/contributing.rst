Contributing
============

Install the development environment from the repository root using Poetry:

.. code-block:: bash

   poetry install --with dev

Keep changes focused, include relevant tests for behavior changes, and update
docstrings and user documentation when public behavior changes.

Run the project's checks before opening a pull request:

.. code-block:: bash

   poetry run black . --check
   poetry run isort . --check
   poetry run pytest

For documentation setup, local previews, and GitHub Pages publishing, see
:doc:`publishing`. Documentation tools are maintained in
``docs/requirements.txt``.

MLflow compatibility checks
---------------------------

CI runs the complete test suite on Python 3.10–3.13 against both MLflow 3.0.1
and the latest stable 3.x release. To reproduce either case, use a disposable
development environment and install the selected MLflow version after the
project dependencies:

.. code-block:: bash

   poetry run pip install "mlflow==3.0.1"
   poetry run pip check
   poetry run pytest

For the latest supported version, replace the first command with
``poetry run pip install --upgrade "mlflow>=3.0.1,<4"``. Record the resolved
MLflow version with the results. Logging integration tests create private SQLite
databases and local artifacts; they do not connect to a configured tracking
server. They verify classification and regression models, preprocessing,
evaluation and tuning entry points, model signatures, artifacts, and run cleanup.

Release metadata
----------------

Package metadata and the release version live in ``pyproject.toml``.
Record release changes in ``CHANGELOG.md`` and migration steps in
``docs/upgrading.md``.

Report bugs or propose improvements through the
`issue tracker <https://github.com/MenaWANG/mlarena/issues>`_.
