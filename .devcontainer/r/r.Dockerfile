# Stage 1: Build and cache R packages
FROM rocker/r-ver:latest AS package-builder

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

# Install renv and setup cache directory
# https://stackoverflow.com/questions/52284345/how-to-show-r-graph-from-visual-studio-code
ENV RENV_PATHS_CACHE=/workspaces/.r/cache
RUN mkdir -p /workspaces/.r/cache && \
    R -e "options(Ncpus = parallel::detectCores()-1); install.packages(c('renv', 'httpgd', 'languageserver'), repos='https://cloud.r-project.org/')"

# Copy requirements file early to leverage caching
COPY rquirements.txt /tmp/rquirements.txt

# Install packages from requirements file using renv
RUN Rscript -e ' \
    options(Ncpus = parallel::detectCores()-1); \
    print(paste("Using RENV_PATHS_CACHE:", Sys.getenv("RENV_PATHS_CACHE"))); \
    pkg <- readLines("/tmp/rquirements.txt"); \
    pkg <- pkg[!grepl("^#", pkg) & pkg != ""]; \
    if (length(pkg) > 0) { \
        print(paste("Installing packages:", paste(pkg, collapse=", "))); \
        renv::install(pkg, repos="https://cloud.r-project.org/"); \
    } else { \
        print("No packages to install from rquirements.txt"); \
    } \
    '

# Stage 2: Final image
FROM rocker/r-ver:latest

# Copy cached packages and environment from package-builder stage
COPY --from=package-builder /workspaces/.r/cache /workspaces/.r/cache
COPY --from=package-builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library

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

# Set environment variable for renv cache
ENV RENV_PATHS_CACHE=/workspaces/.r/cache

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

WORKDIR /workspaces