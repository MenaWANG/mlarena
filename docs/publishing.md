# Building and publishing documentation

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

## Enable GitHub Pages once

After the documentation changes and `.github/workflows/docs.yml` have been
merged into the repository's default branch:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open **Actions → Documentation → Run workflow** on the default branch.
4. Wait for both the build and deployment jobs to pass. The deployment job
   reports the published URL.

The expected project URL is `https://menawang.github.io/mlarena/`, unless a
custom domain is configured. See GitHub's
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
