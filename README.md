# PoI Clustering with Stay Points/Stay Region Detection

Python implementation of PoI (Point-of-Interest) clustering algorithm based on:
* Yang Ye, Yu Zheng, Yukun Chen, Jianhua Feng, and Xing Xie. 2009. Mining Individual Life Pattern Based on Location History. In Proceedings of the 2009 Tenth International Conference on Mobile Data Management: Systems, Services and Middleware (MDM '09). http://dx.doi.org/10.1109/MDM.2009.11
* Raul Montoliu, Jan Blom, and Daniel Gatica-Perez. 2013. Discovering Places of Interest in Everyday Life from Smartphone Data. Multimedia Tools And Applications, 62, 1, 179-207. http://dx.doi.org/10.1007/s11042-011-0982-z
* Vincent W. Zheng, Yu Zheng, Xing Xie, and Qiang Yang. 2010. Collaborative Location and Activity Recommendations with GPS History Data. In Proceedings of the 19th International Conference on World Wide Web (WWW '10). https://doi.org/10.1145/1772690.1772795.


To find PoI, those studies propose a *stay point*, that is a micro cluster of temporal-spatial trajectories, and a *stay region*, that is a macro cluster of stay points. 

## Simple Description of Algorithm

* Find stay points regarding temporal and spatial distance between two trajectories. 
* Build grids that embodying stay points.
* Cluster neighboring grids and give labels. 

## Installation
```cmd
pip install poi-clustering
```

## How to Use
This implementation follows scikit-learn's grammar; *fit* and *predict*. For more details, please see docstrings in codes. 

```commandline
pip install poi-clustering kse801-dataset 
```

```python
from kse801.abc import load_location
import numpy as np
from poi import PoiCluster

# Dummy data of gps coordinates
loc = load_location()
loc_deg = loc.loc[:, ['latitude', 'longitude']].to_numpy()
loc_rad = np.radians(loc_deg)
timestamps = loc.loc[:, 'timestamp'].values
cluster = PoiCluster(d_max=250, r_max=500, t_max=60, t_min=5)
cluster.fit(x=loc_rad, timestamps=timestamps)
label = cluster.predict(loc_rad)
```

