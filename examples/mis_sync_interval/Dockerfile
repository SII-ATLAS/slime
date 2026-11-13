# syntax=docker/dockerfile:1.7
ARG SLIME_IMAGE
FROM ${SLIME_IMAGE}
ARG SLIME_COMMIT
COPY --from=slime / /opt/slime-coop/
WORKDIR /opt/slime-coop
RUN python3 -m pip install --no-deps --no-build-isolation . && \
    printf '%s\n' "$SLIME_COMMIT" > /opt/slime-coop/COMMIT
LABEL org.opencontainers.image.source="https://github.com/SII-ATLAS/slime" \
      org.opencontainers.image.revision="${SLIME_COMMIT}"
