#!/usr/bin/env bash
# Source this file to activate the project's local dev environment:
#   source activate_env.sh
#
# WHY THIS EXISTS (read once, then forget about it):
# This machine is an Intel Mac (x86_64) but shipped with Python 3.13.
# PyTorch dropped macOS x86_64 wheels after 2.2.2, and torch 2.2.2 only
# supports Python 3.10/3.11 -- so torch could not be installed under the
# system Python 3.13 at all. Homebrew's python@3.11 formula had no
# precompiled bottle for this OS version and was compiling everything
# (including OpenSSL/xz test suites) from source, which was far too slow
# on this CPU. Instead, Python 3.11.9 was extracted directly from the
# official python.org installer .pkg (via `pkgutil --expand-full`, no
# sudo/install needed) into .pytools/Python.framework, and two of its
# Mach-O binaries had a hardcoded load path
# (/Library/Frameworks/Python.framework/...) rewritten with
# install_name_tool to point at the local copy instead. .venv was then
# built from that interpreter. None of this affects Colab, which runs
# its own Linux/CUDA Python -- it's purely a local-dev workaround.
#
# torch is pinned to 2.2.2 (last macOS-x86_64 wheel) and transformers /
# sentence-transformers are pinned to versions that still support
# torch<2.4 (newer releases hard-require torch>=2.4). numpy is pinned
# <2 because torch 2.2.2 was built against the numpy 1.x ABI. PennyLane
# prints a deprecation warning about numpy<2 but works fine in practice.

export DYLD_FALLBACK_FRAMEWORK_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.pytools"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/activate"
echo "[activate_env.sh] venv active: $(python --version), DYLD_FALLBACK_FRAMEWORK_PATH set."
