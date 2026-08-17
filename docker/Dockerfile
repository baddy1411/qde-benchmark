# Reproduction environment for the QDE benchmark.
#
# WHY THIS EXISTS
# The thesis reports byte-identical reproduction of every committed artifact, but
# only on the machine that produced them: a single Apple-silicon laptop, where
# NumPy links against Apple's Accelerate framework. Cross-platform floating-point
# behaviour was declared untested. This image pins the userspace -- interpreter,
# every package version, and the BLAS binary -- so that claim can be separated
# into two: repeatability (same environment, same bits) and portability (a
# different BLAS on a different architecture).
#
# Expect the numerical answer to be "close but not bit-identical" across BLAS
# implementations. That is the honest finding, not a defect, and docker/README.md
# records what was actually measured rather than what would be convenient.
#
# DETERMINISM INSIDE THE IMAGE
# Threaded BLAS reduces in a non-deterministic order, so thread counts are pinned
# to 1 below. That makes runs inside this image repeatable; it also makes them
# slower, which is the correct trade for a verification environment.

# Pinned by digest: tags are mutable, so `python:3.9-slim-bookworm` would silently
# become a different image over time. This digest was resolved locally against the
# image this project was actually built and verified with (linux/arm64,
# Python 3.9.25). Re-resolve deliberately with `make docker-pin`.
FROM python@sha256:a02e9c5406c416c504d6c9a1a306ff4080c3173f1008d192f953bd20382a2d5c AS base

# Interpreter version is part of the environment identity; fail loudly rather
# than silently reproducing under a different minor release.
ARG EXPECTED_PYTHON=3.9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # single-threaded BLAS: fixes reduction order so results are repeatable
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    # matplotlib must not try to open a display or write a cache into $HOME
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

# libgomp is required by lightgbm and xgboost at import time; git is used by the
# provenance helper to record the commit a run was executed from.
RUN set -eux; \
    python -c "import sys; assert '.'.join(map(str, sys.version_info[:2])) == '${EXPECTED_PYTHON}', sys.version" ; \
    apt-get update; \
    apt-get install -y --no-install-recommends libgomp1 git; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dependencies are installed before the source is copied so that editing code
# does not invalidate the (slow) dependency layer.
COPY docker/requirements.txt /tmp/requirements.txt

# torch from the CPU index; everything else from PyPI. --extra-index-url would
# let pip pick either source per package, which is exactly the ambiguity a lock
# is supposed to remove, so torch is installed on its own first.
#
# No process substitution here: RUN uses /bin/sh, which is dash on Debian, and
# dash has no <(...).
#
# pip is upgraded (and pinned) first. The pip shipped in python:3.9-slim rejects
# the PyTorch index's `typing_extensions` wheels with "has inconsistent Name:
# expected 'typing-extensions', but metadata has 'typing_extensions'" — it
# compares the metadata name without normalising it. Because --index-url confines
# pip to that one index, it cannot fall back to PyPI and the build dies. Newer pip
# normalises both sides.
RUN set -eux; \
    pip install --upgrade "pip==25.0.1" "setuptools==75.3.0" "wheel==0.45.1"; \
    grep -E '^torch==' /tmp/requirements.txt > /tmp/torch.txt; \
    grep -vE '^torch==' /tmp/requirements.txt > /tmp/rest.txt; \
    pip install --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch.txt; \
    pip install -r /tmp/rest.txt

# Record the resolved transitive closure inside the image. This is the artifact
# that environment_id hashes, and it is what `make docker-lock` exports.
RUN pip freeze --all > /opt/requirements.lock.txt

COPY . /work
# Installed non-editable, then PYTHONPATH puts /work first. That way the image is
# self-contained when run with no mount, and a bind-mounted host checkout at /work
# transparently takes precedence when there is one.
RUN pip install --no-deps .
ENV PYTHONPATH=/work

# Run as a non-root user, and make it own the mounted workspace paths it writes.
RUN useradd --create-home --uid 1000 qde && chown -R qde:qde /work /opt/requirements.lock.txt
USER qde

# Fail the build if the image cannot import the package or see its own lockfile.
RUN python -c "import qdepipe, numpy, pandas, torch, lightgbm, xgboost; print('import ok')" \
 && test -s /opt/requirements.lock.txt

LABEL org.opencontainers.image.title="qde-repro" \
      org.opencontainers.image.description="Pinned reproduction environment for the QDE quantum/classical reservoir benchmark" \
      org.opencontainers.image.source="https://github.com/baddy1411/qde"

CMD ["python", "-c", "import qdepipe, sys, numpy; print('qde reproduction image'); print('python', sys.version.split()[0]); numpy.__config__.show()"]
