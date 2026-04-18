#!/usr/bin/env bash
set -euo pipefail

# Builds a Python wheel from the current source directory using cibuildwheel.
# Mirrors the linux build in pypi-build/.github/workflows/build.yml.
#
# Usage: ./build_wheel.sh [--python <tag>]
#   --python  cibuildwheel build selector, e.g. cp311-manylinux_x86_64
#             Defaults to cp311-manylinux_x86_64

PYTHON_BUILD="313"
DEPS_TAG="v19.0.0"

while [[ $# -gt 0 ]]; do
  case $1 in
    --python) PYTHON_BUILD="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install cibuildwheel --quiet

CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_34 \
CIBW_BUILD="cp$PYTHON_BUILD-manylinux_x86_64" \
CIBW_SKIP="*-musllinux* *-manylinux_i686" \
CIBW_BEFORE_ALL_LINUX="dnf install -y bc ccache && bash {package}/CI/dependencies/setup.sh -t ${DEPS_TAG} -d deps -e env.sh" \
CIBW_ENVIRONMENT_LINUX="CMAKE_PREFIX_PATH=\$PWD/deps/venv:\$PWD/deps/view" \
CIBW_BEFORE_BUILD="ccache -z" \
CIBW_BEFORE_TEST="ccache -s" \
CIBW_TEST_COMMAND="python {project}/Examples/Scripts/Python/track_finding_python_only.py" \
  python -m cibuildwheel --platform linux "$SCRIPT_DIR"

echo ""
echo "Wheel written to: $SCRIPT_DIR/wheelhouse/"
ls "$SCRIPT_DIR/wheelhouse/"*.whl 2>/dev/null || true
