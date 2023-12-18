import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from math import sqrt, ceil, pi
from numpy.linalg import norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def featscale(ser: pd.Series) -> pd.Series:
    return (ser - ser.min()) / (ser.max() - ser.min())

def get_plotting_frame(df: pd.DataFrame, featcols: list, clustercol: str) -> tuple[pd.DataFrame, DataFrameGroupBy, list]:
    plotting_frame = df[featcols].apply(featscale)
    cluster_groups = plotting_frame.groupby(df[clustercol])

    dist_series = []
    centroids = []

    for _, group in cluster_groups:
        centroid = group.mean()
        centroids.append(centroid)
        dist_series.append(pd.Series(norm(group - centroid, axis=1), index=group.index))

    plotting_frame['cluster'] = df[clustercol]
    plotting_frame['d'] = pd.concat(dist_series)

    plotting_frame = plotting_frame.sort_values(['cluster','d'])
    plotting_frame['intra_group_order'] = plotting_frame.groupby('cluster').cumcount() # can't use previous groupby bc order has changed
    plotting_frame = plotting_frame.sort_values(by=['intra_group_order', 'cluster'])

    return plotting_frame, cluster_groups, centroids

def get_subspace(centroids: list, cluster_groups: DataFrameGroupBy, method='tsne', **kwargs) -> pd.DataFrame:
    perplexity = kwargs.get('perplexity', 20)
    early_exaggeration = kwargs.get('early_exaggeration', 15)
    learning_rate = kwargs.get('learning_rate', 400)

    if method == 'tsne':
        xy = TSNE(perplexity=perplexity,
                  early_exaggeration=early_exaggeration,
                  learning_rate=learning_rate).fit_transform(centroids)
        
    elif method == 'pca':
        xy = PCA(n_components=2).fit_transform(centroids)
    
    subspace = pd.DataFrame(xy)
    subspace.columns = ['x','y']
    subspace['cluster'] = cluster_groups.groups.keys()

    return subspace

def bin_subspace(subspace: pd.DataFrame, cluster_groups: DataFrameGroupBy, spread_factor = 1) -> pd.DataFrame:
    rangex = subspace.x.max() - subspace.x.min()
    rangey = subspace.y.max() - subspace.y.min()

    num_clusters_along_side = ceil(sqrt(len(cluster_groups)))
    average_cluster_size = ceil(cluster_groups.size().mean())
    r_avg = ceil(sqrt((average_cluster_size / pi)))

    if rangex > rangey:
        num_bins_y = ceil( r_avg * 2 * num_clusters_along_side * spread_factor )
        num_bins_x = ceil( rangex / rangey * num_bins_y * spread_factor) 
    elif rangey > rangex:
        num_bins_x = ceil( r_avg * 2 * num_clusters_along_side * spread_factor )
        num_bins_y = ceil( rangey / rangex * num_bins_x * spread_factor)

    subspace['x_bin'] = pd.cut(subspace['x'],num_bins_x,labels=False,include_lowest=True)
    subspace['y_bin'] = pd.cut(subspace['y'],num_bins_y,labels=False,include_lowest=True)

    subspace['xybin'] = [str(subspace.x_bin.loc[i]) + '_' + str(subspace.y_bin.loc[i]) for i in subspace.index]

    assert subspace.xybin.value_counts().max()==1, 'Multiple cluster centers in same bin. Increase spread_factor.'

    return subspace

def _circle_generator(center_x: int, center_y: int) -> tuple[int,int]:
    r = 1
    while True:
        points_at_current_radius = []
        
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                dist_sq = dx**2 + dy**2
                if r**2 <= dist_sq < (r+1)**2:
                    points_at_current_radius.append((center_x + dx, center_y + dy))

        if not points_at_current_radius:
            r += 1
            continue

        for pt in points_at_current_radius:
            yield pt
        
        r += 1

def grow_entourages(plotting_frame: pd.DataFrame, subspace: pd.DataFrame) -> pd.DataFrame:
    gridpoint_generators = {}

    for i in subspace.index:
        cluster_label = subspace.cluster.loc[i]
        x = subspace.x_bin.loc[i]
        y = subspace.y_bin.loc[i]
        
        gridpoint_generators[cluster_label] = _circle_generator(x,y)

    occupied_positions = set()
    pts = []
    for i in plotting_frame.index:
        cluster_label = plotting_frame.cluster.loc[i]
        while True:
            candidate_point = next(gridpoint_generators[cluster_label])
            if candidate_point not in occupied_positions:
                pts.append(candidate_point)
                occupied_positions.add(candidate_point)
                break

    return pts