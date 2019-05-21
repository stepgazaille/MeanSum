# Use official pytorch image
FROM pytorch/pytorch:0.4_cuda9_cudnn7

# Set the working directory to /wikisum
WORKDIR /home/meansum

# Copy scripts directory to container WORKDIR
COPY requirements.txt requirements.txt
COPY install_python_pkgs.sh install_python_pkgs.sh
COPY update_tensorboard.py update_tensorboard.py

# Install dependencies:
RUN pip install --upgrade pip
RUN bash install_python_pkgs.sh
RUN python update_tensorboard.py

# Cleanup:
RUN rm requirements.txt
RUN rm install_python_pkgs.sh
RUN rm update_tensorboard.py 

