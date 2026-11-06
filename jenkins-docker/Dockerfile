
FROM jenkins/jenkins:lts-jdk21

USER root

# Install Python and virtual environment support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

# Create Python virtual environment
RUN python3 -m venv /opt/venv

RUN chown -R jenkins:jenkins /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

USER jenkins
