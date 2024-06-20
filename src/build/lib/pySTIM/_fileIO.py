import os
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from typing import Any
from tqdm import tqdm
from scipy.sparse import csr_matrix

def load_xenium(xenium_path: str) -> Any:
    """
    Load Xenium data into an AnnData object.
    
    Parameters:
    - xenium_path (str): Path to the Xenium output directory.

    Returns:
    - AnnData.
    """
    
    xenium_adata = sc.read_10x_h5(os.path.join(xenium_path, 'cell_feature_matrix.h5'))

    cell_coord = pd.read_csv(os.path.join(xenium_path, 'cells.csv.gz'), compression='gzip')
    molecule_data = pd.read_csv(os.path.join(xenium_path, 'transcripts.csv.gz'), compression='gzip')
    cell_seg = pd.read_csv(os.path.join(xenium_path, 'cell_boundaries.csv.gz'), compression='gzip')
    xenium_umap = pd.read_csv(os.path.join(xenium_path, 'analysis/umap/gene_expression_2_components/projection.csv'))
    xenium_clusters = pd.read_csv(os.path.join(xenium_path, 'analysis/clustering/gene_expression_graphclust/clusters.csv'))

    # Replace cell IDs that are -1 with 0
    molecule_data.loc[molecule_data['cell_id'] == -1, 'cell_id'] = 0
    
    xenium_adata.obs = cell_coord.copy()

    # Filter cells based on UMAP results
    cell_id_keep = xenium_umap.Barcode.tolist()
    xenium_adata = xenium_adata[xenium_adata.obs.cell_id.isin(cell_id_keep)]
    cell_seg = cell_seg[cell_seg.cell_id.isin(cell_id_keep)]

    # Update spatial information
    xenium_adata.obsm["spatial"] = xenium_adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

    # Update UMAP data
    xenium_umap.index = xenium_adata.obs_names
    xenium_adata.obsm['X_umap'] = xenium_umap.iloc[:, 1:3].copy().to_numpy()

    # Update cluster information
    clusters = xenium_clusters['Cluster'].to_list()
    xenium_adata.obs['leiden'] = list(map(str, clusters))

    gene_names = xenium_adata.var_names.astype(str).tolist()
    molecule_data = molecule_data[molecule_data.feature_name.isin(gene_names)]
    molecule_data = molecule_data[molecule_data.cell_id.isin(cell_id_keep)]

    xenium_adata.uns['seg'] = cell_seg
    xenium_adata.uns['transcript'] = molecule_data
    xenium_adata.uns['datatype'] = "xenium"

    return xenium_adata

def load_xenium_baysor(data_dir):
    """
    Load Baysor segmented Xenium data.

    Parameters:
    - data_dir (str): Path to the Baysor output directory.

    Returns:
    - two Anndata object, one saves cells, the other saves polygons.
    """

    print("Loading baysor segmentation output...")
    counts = pd.read_csv(os.path.join(data_dir, "segmentation_counts.tsv"), sep="\t")

    cells_meta = pd.read_csv(os.path.join(data_dir, "segmentation_cell_stats.csv"))
    cells_meta.index = cells_meta.cell

    molecule_data = pd.read_csv(os.path.join(data_dir, "segmentation.csv"))
    molecule_data = molecule_data[molecule_data["cell"] != 0]
    molecule_data["x_location"] = molecule_data["x"] * 0.2125
    molecule_data["y_location"] = molecule_data["y"] * 0.2125
    
    features = list(counts["gene"])
    counts = csr_matrix(counts.iloc[:, 1:].values, dtype=np.int32)
    adata = anndata.AnnData(counts.T, obs=cells_meta)
    adata.var_names = features
    adata.uns["datatype"] = "xenium"

    adata.obs["x_centroid"] = adata.obs["x"] * 0.2125 ## convert pixel to micron
    adata.obs["y_centroid"] = adata.obs["y"] * 0.2125
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
    
    print("Loading polygons...")
    polygon_df = pd.read_csv(os.path.join(data_dir, "polygon.csv"))
    polygon_meta = pd.read_csv(os.path.join(data_dir, "polygon_meta.csv"))
    polygon_count = pd.read_csv(os.path.join(data_dir, "polygon_count.csv"))
    polygon_df["x_centroid"] = polygon_df["x_location"] * 0.2125
    polygon_df["y_centroid"] = polygon_df["y_location"] * 0.2125

    poly_counts = csr_matrix(polygon_count.iloc[:, 1:].values, dtype=np.int32)
    polygon_centroid = polygon_df.groupby('polygon_number').mean().reset_index()
    polygon_meta = polygon_meta.merge(polygon_centroid, on="polygon_number", how="inner")
    polygon_meta.index = polygon_meta.polygon_number.astype(str)
    poly_adata = anndata.AnnData(poly_counts, obs=polygon_meta)
    poly_adata.var_names = list(polygon_count.columns[1:])

    poly_adata.uns["seg"] = polygon_df
    poly_adata.uns["transcript"] = molecule_data

    print("Creating polygons...")
    seg = poly_adata.uns['seg']
    grouped = seg.groupby("polygon_number")
    all_polygons = sorted(list(set(seg["polygon_number"])))
    new_lib = [{
        'coordinates': [grouped.get_group(cell).iloc[:, 3:5].to_numpy().tolist()],
        'type': 'Polygon'
    } for cell in tqdm(all_polygons)]

    new_poly = dict(zip(map(str, all_polygons), new_lib))
    poly_adata.uns['poly'] = new_poly

    return adata, poly_adata

def load_polygon(data_dir):

    print("Loading baysor segmentation output...")
    molecule_data = pd.read_csv(os.path.join(data_dir, "segmentation.csv"))
    molecule_data = molecule_data[molecule_data["cell"] != 0]
    molecule_data["x_location"] = molecule_data["x"] * 0.2125
    molecule_data["y_location"] = molecule_data["y"] * 0.2125
    print("Loading polygons...")
    polygon_df = pd.read_csv(os.path.join(data_dir, "polygon.csv"))
    polygon_meta = pd.read_csv(os.path.join(data_dir, "polygon_meta.csv"))
    polygon_count = pd.read_csv(os.path.join(data_dir, "polygon_count.csv"))
    polygon_df["x_centroid"] = polygon_df["x_location"] * 0.2125
    polygon_df["y_centroid"] = polygon_df["y_location"] * 0.2125

    poly_counts = csr_matrix(polygon_count.iloc[:, 1:].values, dtype=np.int32)
    polygon_centroid = polygon_df.groupby('polygon_number').mean().reset_index()
    polygon_meta = polygon_meta.merge(polygon_centroid, on="polygon_number", how="inner")
    polygon_meta.index = polygon_meta.polygon_number.astype(str)
    poly_adata = anndata.AnnData(poly_counts, obs=polygon_meta)
    poly_adata.var_names = list(polygon_count.columns[1:])

    poly_adata.uns["seg"] = polygon_df
    poly_adata.uns["transcript"] = molecule_data

    print("Creating polygons...")
    seg = poly_adata.uns['seg']
    grouped = seg.groupby("polygon_number")
    all_polygons = sorted(list(set(seg["polygon_number"])))
    new_lib = [{
        'coordinates': [grouped.get_group(cell).iloc[:, 3:5].to_numpy().tolist()],
        'type': 'Polygon'
    } for cell in tqdm(all_polygons)]

    new_poly = dict(zip(map(str, all_polygons), new_lib))
    poly_adata.uns['poly'] = new_poly

    return  poly_adata

    
