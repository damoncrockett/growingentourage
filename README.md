# Growing Entourage Plot
A technique for zero occlusion unit visualization of clusters of points (usually images or glyphs). Journal article [here]([https://dahj.org/article/direct-visualization-techniques](https://journals.ub.uni-heidelberg.de/index.php/dah/article/view/33529)).

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

pts = grow_entourages(plotting_frame, subspace)

plotting_frame['x'] = [item[0] for item in pts]
plotting_frame['y'] = [item[1] for item in pts]

# ivpy plot

sys.path.append(os.path.expanduser("~") + "/ivpy/src")
from ivpy import *

imagecol = '...'

attach(plotting_frame, imagecol)
scatter('x','y') 
```
