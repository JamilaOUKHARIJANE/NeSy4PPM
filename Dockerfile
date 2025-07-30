FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Set environment variable
ENV NAME=Nesy4ppm

# Install Graphviz system dependencies
RUN apt-get update && \
    apt-get install -y graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r docs/source/requirements.txt
RUN pip install -e .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Notebook server
CMD ["jupyter", "notebook", "--notebook-dir=/app", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]