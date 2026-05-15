FROM condaforge/miniforge3:24.1.2-0

# Install system build tools (required for pip to compile packages like openslide-python)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy project files and set working directory 
WORKDIR /app

COPY . /app/NBT-Classifier

WORKDIR /app/NBT-Classifier

# Create env
RUN mamba env create -f environment.yml

# Copy custom HistoQC config so it can be used via shorthand (e.g., "histoqc -c NBT")
RUN CONFIG_DIR=$(conda run -n nbtclassifier python -c "import histoqc; from pathlib import Path; print(Path(histoqc.__file__).parent / 'config')") && \
    echo "CONFIG_DIR=$CONFIG_DIR" && \
    cp config/config_NBT.ini "$CONFIG_DIR"/

# Clean caches
RUN mamba clean -a -y

# Auto-activate environment in interactive shells
SHELL ["bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate nbtclassifier" >> ~/.bashrc

# Automatically activate env in container sessions
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nbtclassifier"]

# Default command
CMD ["/bin/bash"]