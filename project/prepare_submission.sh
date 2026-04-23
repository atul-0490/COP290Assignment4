#!/usr/bin/env bash
# Clean the project tree, render report.pdf (pandoc), verify every required
# file is present, and package the submission tar that Moodle expects.
set -e

ENTRY=${1:-}
if [ -z "$ENTRY" ]; then
    echo "Usage: ./prepare_submission.sh <entry_number>"
    echo "  e.g. ./prepare_submission.sh 2025ANZ0000"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# 1. Remove build artifacts so the tar is source-only.
# -----------------------------------------------------------------------------
echo "==> Cleaning build artifacts"
# Remove every build-flavour directory (build, build-asan, build-release, ...)
# plus stray object / archive files.
for d in build build-*; do
    [ -d "$d" ] && rm -rf "$d"
done
find . -name "*.o" -delete
find . -name "*.a" -delete
# Stale PNGs / DOT files from explain() rehearsals.
find . -maxdepth 2 -name "*.dot" -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# 2. Render report.md → report.pdf (best effort; pandoc is optional).
# -----------------------------------------------------------------------------
echo "==> Rendering report.pdf"
if command -v pandoc >/dev/null 2>&1; then
    if pandoc report.md -o report.pdf 2>/tmp/pandoc_err; then
        echo "    report.pdf written."
    else
        echo "    pandoc failed (details in /tmp/pandoc_err)."
        echo "    Manually convert report.md to report.pdf before submitting."
    fi
else
    echo "    WARNING: pandoc not installed."
    echo "    Manually convert report.md to report.pdf before submitting."
fi

# -----------------------------------------------------------------------------
# 3. Verify every required file is present.
# -----------------------------------------------------------------------------
echo "==> Verifying required files"
REQUIRED=(
    "CMakeLists.txt"
    "README.md"
    "report.md"
    "main.cpp"
    "include/DataFrame.hpp"
    "include/EagerDataFrame.hpp"
    "include/LazyDataFrame.hpp"
    "include/QueryOptimizer.hpp"
    "include/Expr.hpp"
    "include/LogicalPlan.hpp"
    "include/IO.hpp"
    "include/TypeUtils.hpp"
    "src/EagerDataFrame.cpp"
    "src/LazyDataFrame.cpp"
    "src/QueryOptimizer.cpp"
    "src/Expr.cpp"
    "src/LogicalPlan.cpp"
    "src/IO.cpp"
    "src/TypeUtils.cpp"
    "tests/test_main.cpp"
)

MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ ! -f "$f" ]; then
        echo "    MISSING: $f"
        MISSING=1
    fi
done

# report.pdf is expected — warn (but do not fail) if it is missing, so
# students without pandoc can still package the tar from a machine that
# has pandoc after the fact.
if [ ! -f "report.pdf" ]; then
    echo "    WARNING: report.pdf is missing."
    echo "             Install pandoc and rerun, or render report.md by hand."
fi

if [ "$MISSING" -eq 1 ]; then
    echo "==> Submission incomplete. Fix missing files before packaging."
    exit 1
fi
echo "    All required files present."

# -----------------------------------------------------------------------------
# 4. Package. Tar from one level above so the archive root is `project/`.
# -----------------------------------------------------------------------------
echo "==> Packaging ${ENTRY}.tar"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
cd "$PARENT_DIR"
tar -cf "${ENTRY}.tar" "${PROJECT_NAME}"

echo ""
echo "Created: ${PARENT_DIR}/${ENTRY}.tar"
echo "Submit this file on Moodle."
