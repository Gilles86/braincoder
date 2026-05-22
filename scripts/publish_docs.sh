#!/usr/bin/env bash
#
# publish_docs.sh — build the sphinx-gallery docs and push them to
# the public GitHub Pages site at https://braincoder-devs.github.io.
#
# The Pages site is served from a *separate* repo
# (braincoder-devs/braincoder-devs.github.io) that holds only the
# rendered HTML. There is no CI doing this automatically, so this
# script is the canonical "publish docs" path.
#
# Usage
# -----
#   bash scripts/publish_docs.sh                # build + push
#   bash scripts/publish_docs.sh --dry-run      # build, stage, show diff, don't push
#   bash scripts/publish_docs.sh --skip-build   # reuse docs/_build/html, just push
#   bash scripts/publish_docs.sh --help
#
# Requirements
# ------------
#   - A working sphinx + sphinx-gallery install (use the braincoder
#     conda env; this also runs every gallery example end-to-end, so
#     have GPU/CPU runtime available).
#   - SSH push access to braincoder-devs/braincoder-devs.github.io.
#   - Run from the repo root or anywhere — the script resolves its
#     own location.
#
# What it does
# ------------
#   1. (unless --skip-build) sphinx-build docs/ → docs/_build/html.
#   2. Clones the Pages repo to a tempdir.
#   3. Wipes everything in the tempdir except .git, .nojekyll, CNAME.
#   4. Copies the fresh build over.
#   5. Stages, commits with a message referencing the source SHA,
#      and pushes (unless --dry-run).
#
# Idempotent. The tempdir clone is removed on exit.

set -euo pipefail

# ----------------------------------------------------------------- args
DRY_RUN=0
SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help)
      sed -n '2,/^# Idempotent/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "run with --help to see usage" >&2
      exit 2
      ;;
  esac
done

# ----------------------------------------------------------------- paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/docs/_build/html"
PAGES_REPO="git@github.com:braincoder-devs/braincoder-devs.github.io.git"

cd "$REPO_ROOT"

# ----------------------------------------------------------------- build
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "==> Cleaning docs/_build/ and rebuilding (sphinx-gallery will execute every example)…"
  rm -rf docs/_build
  # -a: write all output files, even unchanged.
  # -E: rebuild environment from scratch.
  sphinx-build -a -E -b html docs "$BUILD_DIR"
else
  echo "==> --skip-build: reusing existing $BUILD_DIR"
  if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: $BUILD_DIR does not exist; remove --skip-build to build it." >&2
    exit 1
  fi
fi

# ----------------------------------------------------------------- clone pages repo
TMP_DIR="$(mktemp -d -t braincoder-docs-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
echo "==> Cloning $PAGES_REPO → $TMP_DIR"
git clone --depth 1 "$PAGES_REPO" "$TMP_DIR/pages" >/dev/null

PAGES_DIR="$TMP_DIR/pages"
cd "$PAGES_DIR"

# Preserve anything outside the docs build that must survive across pushes.
# CNAME (custom domain) and .nojekyll (tells GH Pages "this is raw HTML —
# don't run Jekyll, allow files starting with _") are the two we know about.
mkdir -p "$TMP_DIR/keep"
for f in .nojekyll CNAME; do
  [[ -f "$PAGES_DIR/$f" ]] && cp "$PAGES_DIR/$f" "$TMP_DIR/keep/"
done

# Wipe tracked content (but keep .git).
echo "==> Wiping old HTML"
find "$PAGES_DIR" -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +

# Restore the preserved bits, then copy the new build over.
for f in .nojekyll CNAME; do
  [[ -f "$TMP_DIR/keep/$f" ]] && mv "$TMP_DIR/keep/$f" "$PAGES_DIR/"
done

echo "==> Copying $BUILD_DIR → $PAGES_DIR"
# trailing slash on src copies *contents* of html/ (not the html/ dir itself)
cp -R "$BUILD_DIR/." "$PAGES_DIR/"

# .nojekyll is mandatory for sphinx output on GH Pages (sphinx uses
# underscore-prefixed dirs like _static/, _images/ which Jekyll hides).
touch "$PAGES_DIR/.nojekyll"

# ----------------------------------------------------------------- commit + push
cd "$PAGES_DIR"
git add -A

if git diff --cached --quiet; then
  echo "==> No changes to publish. Exiting."
  exit 0
fi

SOURCE_SHA="$(cd "$REPO_ROOT" && git rev-parse --short=12 HEAD)"
SOURCE_BRANCH="$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD)"
MSG="docs: rebuild from Gilles86/braincoder@${SOURCE_SHA} (${SOURCE_BRANCH})"

echo "==> Diff summary:"
git diff --cached --stat | tail -10

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "==> --dry-run: staged but not committing/pushing."
  echo "    Build is at: $BUILD_DIR"
  echo "    Pages clone (staged) at: $PAGES_DIR  (will be wiped on exit)"
  exit 0
fi

echo "==> Committing: $MSG"
git commit -m "$MSG"

echo "==> Pushing to braincoder-devs/braincoder-devs.github.io"
git push origin HEAD

echo
echo "✓ Published. Allow ~30 s for GitHub Pages to refresh, then check:"
echo "  https://braincoder-devs.github.io/"
echo "  https://braincoder-devs.github.io/auto_examples/01_decoding_pipeline/"
