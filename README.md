# scVISA

scVISA adds self-attention mechanism with a scalable residual connection to scVI's Encoder for generating latent embedding. Using a scalable residual self-attention technique allows us to extract biologically meaningful signals from such single-cell expression data for clustering. 

## Installation
Simply clone this repo and use the `requirements.txt` file to install the pre-requisites. Please note that installation of `scvi-tools` through `pip` without creating a seperate enviroment may cause version conflicts. We recommend that you use a Google Colab or Kaggle instance to execute this code. 

On a Google Colab notebook, simply run the following cell to get started.

```python
!git clone https://github.com/rangan2510/scVISA
!pip install -r /content/scVISA/requirements.txt --quiet &> /dev/null
```

When using Kaggle, run the following cell to get started;
```python
!git clone https://github.com/rangan2510/scVISA
!pip install -r ./scVISA/requirements.txt --quiet
```

## Generating latent embeddings
Creating the model can be done simply by importing the model and setting up the model using the SCVI's anndata manager. 

```python
from scVISA.model import scVISA
adata = your adata file
# move the data to a new layer.
adata.layers["counts"] = adata.X.copy()
scVISA.setup_anndata(
    adata,
    layer="counts",
)
model = scVISA(adata)
model.train(10)
```

Once the model is trained, you can simply get the latent embeddings and use scanpy for furhter analysis. 

```python
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent

# process with scanpy
import scanpy as sc
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=20)
sc.tl.umap(adata, min_dist=0.3)
```

## Custering
Clustering is similar to how you use scanpy for clustering.

```python
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, key_added="leiden_scVI", resolution=0.5)

sc.pl.umap(
    adata,
    color=["leiden_scVI"],
    frameon=False,
    save="clustering.pdf"
)
```
## Running on Colab

Download the `demo.ipynb` file and upload it to Google Colab to run scVISA on the cloud. The demo uses the purified PBMC dataset from scvi. You can upload and use your datasets too, but you will have to modify the visualization code accordingly. 