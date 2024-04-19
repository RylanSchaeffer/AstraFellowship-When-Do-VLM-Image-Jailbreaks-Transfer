# To build, navigate to the *root* of this repo and run:
#   docker build . -f docker/Dockerfile -t rylanschaeffer/universal_vlm_jailbreak:main
# You may also want to push the image to Docker Hub:
#   docker push rylanschaeffer/universal_vlm_jailbreak:main
# To play around with the image, you can run the command:
#   docker run -it rylanschaeffer/universal_vlm_jailbreak:main /bin/bash
# Ping Tony if you have trouble pushing, he may need to give you permissions.

#### BEGIN: Load micromamba ####################################################
# We copy micromamba from the micromamba stage because historically the download
# link has been unreliable.
FROM mambaorg/micromamba:1.5.6-jammy as micromamba
#### END: Load micromamba ######################################################


#### BEGIN: Build flash-attn ###################################################
# We use the devel image to build flash-attn.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS flash-attn-build
ARG DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update -q \
  && apt-get install -y --no-install-recommends \
  # Required to build flash-attn
  curl \
  git \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Add defender user
RUN adduser --disabled-password --gecos "" defender

# Add .local/bin to defender PATH. Needed for pip installs.
RUN echo "export PATH=\$PATH:/home/defender/.local/bin" >> /home/defender/.bashrc

# Switch to defender user
USER defender
WORKDIR /home/defender

# Install micromamba
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#manual-installation
COPY --from=micromamba --chown=defender /bin/micromamba .local/bin/micromamba
RUN .local/bin/micromamba shell init -s bash -p /home/defender/.micromamba
RUN .local/bin/micromamba config append channels conda-forge \
  && .local/bin/micromamba config append channels nodefaults \
  && .local/bin/micromamba config set channel_priority strict

# Set up micromamba environment
ARG MAMBA_ROOT_PREFIX=/home/defender/.micromamba
RUN /home/defender/.local/bin/micromamba env create -y -n llmj python=3.11 \
  && /home/defender/.local/bin/micromamba clean --all -y -f
RUN /home/defender/.local/bin/micromamba run -n llmj \
  pip install --no-cache-dir --upgrade pip setuptools

# Install flash-attn dependencies and build flash-attn
# See https://github.com/Dao-AILab/flash-attention for details.
# The torch version here needs to match the one in requirements.txt.
COPY --chown=defender requirements.txt downloads/requirements.txt
RUN /home/defender/.local/bin/micromamba run -n llmj \
  pip install --no-cache-dir \
  ninja==1.11.1.1 \
  packaging \
  $(cat downloads/requirements.txt | grep torch==)
RUN /home/defender/.local/bin/micromamba run -n llmj \
  pip install --user --no-cache-dir flash-attn==2.5.3 --no-build-isolation
#### END: Build flash-attn #####################################################


#### BEGIN: Final image ########################################################
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 as final
ARG DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update -q \
  && apt-get install -y --no-install-recommends \
  # General utilities
  curl \
  git \
  git-lfs \
  htop \
  openssh-server \
  sudo \
  tmux \
  unzip \
  uuid-runtime \
  vim \
  wget \
  # Latex
  texlive-latex-extra \
  texlive-fonts-recommended \
  dvipng \
  cm-super \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Add defender user with sudo privileges
RUN adduser --disabled-password --gecos "" defender
RUN usermod -aG sudo defender
RUN echo "defender ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/defender

# Add .local/bin to defender PATH. Needed for pip installs.
RUN echo "export PATH=\$PATH:/home/defender/.local/bin" >> /home/defender/.bashrc

# Switch to defender user
USER defender
WORKDIR /home/defender

# Set up git
RUN git config --global credential.helper store
RUN git config --global core.editor "vim"

# Install micromamba
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#manual-installation
COPY --from=micromamba --chown=defender /bin/micromamba .local/bin/micromamba
RUN .local/bin/micromamba shell init -s bash -p /home/defender/.micromamba
RUN .local/bin/micromamba config append channels conda-forge \
  && .local/bin/micromamba config append channels nodefaults \
  && .local/bin/micromamba config set channel_priority strict

# Create micromamba environment
ARG MAMBA_ROOT_PREFIX=/home/defender/.micromamba
RUN /home/defender/.local/bin/micromamba env create -y -n llmj python=3.11 \
  && /home/defender/.local/bin/micromamba clean --all -y -f
RUN /home/defender/.local/bin/micromamba run -n llmj \
  pip install --no-cache-dir --upgrade pip setuptools

# Install pip requirements
COPY --chown=defender requirements.txt downloads/requirements.txt
RUN --mount=type=cache,target=/home/defender/.cache,uid=1000,gid=1000 \
  /home/defender/.local/bin/micromamba run -n llmj \
  pip install -r downloads/requirements.txt

# Copy flash-attn from flash-attn-build
COPY --from=flash-attn-build --chown=defender \
   /home/defender/.local/lib/python3.11/site-packages/flash_attn \
   /home/defender/.local/lib/python3.11/site-packages/flash_attn
COPY --from=flash-attn-build --chown=defender \
   /home/defender/.local/lib/python3.11/site-packages/flash_attn-2.5.3.dist-info \
   /home/defender/.local/lib/python3.11/site-packages/flash_attn-2.5.3.dist-info
COPY --from=flash-attn-build --chown=defender \
   /home/defender/.local/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so \
   /home/defender/.local/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so

# Copy start script (for runpod)
COPY --chown=defender docker/start.sh downloads/start.sh
CMD ["downloads/start.sh"]
#### END: Final image ########################################################