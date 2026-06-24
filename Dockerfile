# AGEE — reproducibility Docker image
#
# This image reproduces every reported AGEE number from the stored
# trajectories WITHOUT requiring Ollama or a GPU. To re-run the LLM agent,
# you need Ollama running on the host and `--network=host` at run time.
#
# Build:    docker build -t agee:tkde .
# Run:      docker run --rm -it -v $(pwd):/workspace agee:tkde
# Test:     docker run --rm agee:tkde python main.py
#
FROM python:3.11.6-slim

LABEL maintainer="Murat Gok <murat.gok@yalova.edu.tr>"
LABEL description="Reproducibility image for the AGEE metric (TKDE submission)"

# System dependencies for igraph (leidenalg backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libigraph-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Smoke test on build — runs the synthetic-graph demo, NOT the LLM
RUN python -c "import sys; sys.path.insert(0,'.'); \
    from core.agee import AGEECalculator; \
    import networkx as nx; \
    G = nx.karate_club_graph(); \
    c = AGEECalculator(G, graph_name='karate_smoketest'); \
    print('Smoke test passed:', c.n_communities, 'communities detected')"

# Default: print the reproduced manuscript numbers from stored trajectories
CMD ["python", "-c", "\
import sys; sys.path.insert(0,'.'); \
import pandas as pd; \
df = pd.read_csv('kgqa_experiment/results/kgqa_trajectories.csv'); \
print('Reproduced from stored trajectories:'); \
print(df.groupby('agent').agg(hits=('hit','mean'), agee=('agee','mean'), n=('hit','size')).round(4))"]
