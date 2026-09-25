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

Package metadata and the release version live in ``pyproject.toml``.
Record release changes in ``CHANGELOG.md`` and migration steps in
``docs/upgrading.md``.

Report bugs or propose improvements through the
`issue tracker <https://github.com/MenaWANG/mlarena/issues>`_.
