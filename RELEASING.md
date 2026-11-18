# Releasing braincoder

A release is finished only when **PyPI, GitHub, Zenodo and the documentation site** all show the new version. The documentation site is not updated automatically: step 5 is a separate, manual step, so do not skip it.

Replace `X.Y.Z` with the new version below.

## 1. Bump the version

On a branch, update all of these:

| File | Field |
|---|---|
| `pyproject.toml` | `version` |
| `braincoder/__init__.py` | `__version__` |
| `CITATION.cff` | `version` |
| `.claude-plugin/plugin.json` | `version` |
| `docs/conf.py` | `release` (`X.Y.Z`) and `version` (`X.Y`) |
| `CHANGELOG.md` | New section `## X.Y.Z (YYYY-MM-DD)` |
| `docs/whats_new.md` | New section `## X.Y.Z highlights (YYYY-MM-DD)` |

## 2. Merge

Open a pull request and merge it once the tests pass on all three backends (TensorFlow, JAX and PyTorch).

## 3. Tag the merge commit

```bash
git switch main && git pull
git tag -a vX.Y.Z -m "braincoder X.Y.Z"
git push origin vX.Y.Z
```

## 4. Publish the GitHub release

Use the new `CHANGELOG.md` section as the release notes (`notes.md` below).

```bash
gh release create vX.Y.Z --verify-tag --latest --title "braincoder X.Y.Z" --notes-file notes.md
```

Publishing the release starts `.github/workflows/publish.yml`, which uploads the package to PyPI. Check that the workflow succeeds.

## 5. Publish the documentation

The site at <https://braincoder-devs.github.io> is served from the separate repository `braincoder-devs/braincoder-devs.github.io`. `scripts/publish_docs.sh` builds and pushes it. The build runs every gallery example, so it takes a while.

Build from the tag in a clean worktree, so that uncommitted local changes cannot end up on the site:

```bash
git worktree add --detach ../braincoder-docs-X.Y.Z vX.Y.Z
cd ../braincoder-docs-X.Y.Z
conda activate braincoder
export PYTHONPATH=$PWD KERAS_BACKEND=tensorflow MPLBACKEND=Agg

bash scripts/publish_docs.sh --dry-run      # build only; nothing is pushed
open docs/_build/html/whats_new.html        # check the new version and that the gallery has no failed examples
bash scripts/publish_docs.sh --skip-build   # push the build you just checked

cd - && git worktree remove --force ../braincoder-docs-X.Y.Z
```

## 6. Check the release

- [ ] `pip index versions braincoder` lists `X.Y.Z` as the latest version.
- [ ] <https://github.com/Gilles86/braincoder/releases> shows `vX.Y.Z` as the latest release.
- [ ] <https://braincoder-devs.github.io/whats_new.html> shows `X.Y.Z` in the page title and has its new section.
- [ ] The Zenodo record has a new version: <https://doi.org/10.5281/zenodo.10778413>.
