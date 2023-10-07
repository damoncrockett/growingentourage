# Growing Entourage Plot
A technique for zero occlusion unit visualization of clusters of points (usually images or glyphs). Journal article [here](https://dahj.org/article/direct-visualization-techniques).

# Usage

```python
import pandas as pd
import sys,os
sys.path.append(os.path.expanduser("~") + "/growingentourage")
from geometry import *

df = pd.DataFrame(...)
featcols = [...]
clustercol = '...'

plotting_frame, cluster_groups, centroids = get_plotting_frame(df, featcols, clustercol)

subspace = get_subspace(centroids, cluster_groups)
subspace = bin_subspace(subspace, cluster_groups, spread_factor = 1)

plotting_frame = grow_entourages(plotting_frame, subspace) # adds columns ['x','y']

# ivpy plot

sys.path.append(os.path.expanduser("~") + "/ivpy/src")
from ivpy import *

imagecol = '...'

attach(plotting_frame, imagecol)
scatter('x','y') 
```
