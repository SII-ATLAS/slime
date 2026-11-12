# syntax=docker/dockerfile:1.7
# Supply an immutable, standard slime dependency image and a clean AvaCore Git
# archive as the named avacore build context. No framework packages are modified.
ARG SLIME_IMAGE
FROM ${SLIME_IMAGE} AS ava
ARG AVACORE_REVISION=6d4f0dd2b9e7d7e830445b64e019ae5ad98d759f
COPY --from=avacore / /opt/avacore-source/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_PROJECT_ENVIRONMENT=/opt/ava
RUN uv python install 3.12.12 && \
    cd /opt/avacore-source && \
    uv sync --frozen --no-dev --no-editable --python 3.12.12 && \
    /opt/ava/bin/python -c 'from ava_core.generate.terminal import TerminalBenchTerminus2; from ava_core.rewards.lmaaj import LMaaJ' && \
    printf '%s\n' "$AVACORE_REVISION" > /opt/ava/AVACORE_REVISION
ENV AVA_PYTHON=/opt/ava/bin/python
ENV AVACORE_SOURCE_REVISION=${AVACORE_REVISION}

FROM ava AS runtime
ARG SLIME_COMMIT
COPY --from=slime / /opt/slime-coop/
WORKDIR /opt/slime-coop
RUN python3 -m pip install --no-deps --no-build-isolation . && \
    printf '%s\n' "$SLIME_COMMIT" > /opt/slime-coop/COMMIT
LABEL org.opencontainers.image.source="https://github.com/SII-ATLAS/slime" \
      org.opencontainers.image.revision="${SLIME_COMMIT}"
