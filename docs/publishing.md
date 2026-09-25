# Building and publishing documentation

The documentation is published at
[MLArena documentation](https://menawang.github.io/mlarena/) through GitHub Pages.
It follows the repository's default branch and may include changes newer than
the PyPI release.

## Build locally

Use Python 3.12 for documentation builds. From the repository root, create and
activate a virtual environment, then run:

```bash
python -m pip install -e . -r docs/requirements.txt
python scripts/check_quickstart.py
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

The build imports the actual package to generate API documentation. Missing
dependencies and documentation warnings should be fixed before publishing.
The Quickstart check executes both Python examples directly from
`docs/quickstart.md`, in order with shared imports, and validates their predictions
and metrics. It checks the installed checkout, not the PyPI release. If you add
or restructure the examples, update `scripts/check_quickstart.py` accordingly.

Open `docs/_build/html/index.html`, or preview through a local server:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Visit `http://localhost:8000`. Generated files are ignored by Git.
Edit Markdown/reStructuredText pages or the relevant Python docstrings, then
rebuild. Package versions are read from `pyproject.toml`.

## Publish updates

GitHub Pages is already configured for this repository. To update the site:

1. Edit the documentation or Python docstrings, then run the local checks above.
2. Open a pull request and wait for the documentation and project checks to pass.
3. Merge into the default branch. The **Documentation** workflow builds and
   deploys the updated site automatically.
4. Wait for both jobs to pass, then check the changed pages on the
   [published site](https://menawang.github.io/mlarena/).

To rebuild the current default branch without a new commit, open
**Actions → Documentation → Run workflow** and select the default branch.
A documentation-only update does not require a PyPI release.

## Initial setup for a fork or new repository

These steps are only needed when setting up another repository, or restoring
its Pages configuration. First merge `.github/workflows/docs.yml` into its
default branch, then:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open **Actions → Documentation → Run workflow** on the default branch.
4. Wait for both the build and deployment jobs to pass. The deployment job
   reports the published URL.

Use the URL reported by the deployment job; a fork has its own Pages URL.
See GitHub's
[custom workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
for the publishing setup.

## Automatic checks and deployment

Pull requests run the Quickstart check and build the documentation with warnings
treated as errors. A failed example prevents publishing.
Pushes to `main` or `master` also build it; deployment runs only when that
branch is the repository's default branch. Manual runs follow the same rule.
If you rename the default branch, update the workflow's branch filters.

The build job has read-only repository access. Only the deployment job receives
Pages and identity-token write permissions. The site displays documentation
from the default branch, which may include changes newer than the PyPI release.
