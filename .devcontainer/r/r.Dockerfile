FROM rocker/r-ver:latest
# FROM r-base:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    ca-certificates \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgit2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install renv, essential R packages, and setup cache directory
# https://stackoverflow.com/questions/52284345/how-to-show-r-graph-from-visual-studio-code
ENV RENV_PATHS_CACHE=/workspaces/.r/cache
RUN mkdir -p /workspaces/.r/cache && \
    # chown -R $USERNAME:$USER_GID /workspaces && \
    # Install renv and other essential packages first
    R -e "options(Ncpus = parallel::detectCores()-1); install.packages(c('renv', 'httpgd', 'languageserver'), repos='https://cloud.r-project.org/')"

# Copy requirements file
COPY rquirements.txt /tmp/rquirements.txt

# Install packages from requirements file using renv
# This leverages renv's caching mechanism specified by RENV_PATHS_CACHE
RUN Rscript -e ' \
    options(Ncpus = parallel::detectCores()-1); \
    print(paste("Using RENV_PATHS_CACHE:", Sys.getenv("RENV_PATHS_CACHE"))); \
    pkg <- readLines("/tmp/rquirements.txt"); \
    pkg <- pkg[!grepl("^#", pkg) & pkg != ""]; \
    if (length(pkg) > 0) { \
        print(paste("Installing packages:", paste(pkg, collapse=", "))); \
        # Use renv::install for installation and caching \
        renv::install(pkg, repos="https://cloud.r-project.org/"); \
    } else { \
        print("No packages to install from rquirements.txt"); \
    } \
    '
# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Add NVIDIA GPU support
# UNCOMMENT IF YOU HAVE A GPU
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspaces/${localWorkspaceFolderBasename}