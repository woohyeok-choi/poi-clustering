import numpy as np
from typing import Optional, Tuple, Union, Dict, Iterable
from itertools import product
from sklearn.base import ClusterMixin
from uuid import uuid4

R = 63710088
NON_LABEL = 'NONE'


def latlng_haversine_dist(l1: Iterable[float], l2: Iterable[float]):
    lat1, lng1 = l1
    lat2, lng2 = l2

    d_lat = lat2 - lat1
    d_lng = lng2 - lng1

    d = np.sin(d_lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(d))


def latlng_centroid(coords: Iterable[Iterable[float]]) -> Iterable[float]:
    coords = coords if type(coords) == np.ndarray else np.array(coords)

    assert len(coords.shape) == 2 and coords.shape[1] == 2, \
        '\'coords\' should be a shape of (n_samples, 2), where the second one is formed as latitude and longitude.'
    lat, lng = coords[:, 0].ravel(), coords[:, 1].ravel()

    x = np.mean(np.cos(lat) * np.cos(lng))
    y = np.mean(np.cos(lat) * np.sin(lng))
    z = np.mean(np.sin(lat))

    c_lng = np.arctan2(y, x)
    c_sqr = np.sqrt(x * x + y * y)
    c_lat = np.arctan2(z, c_sqr)

    return c_lat, c_lng


def latlng_eq_rect(ref_coord: Iterable[float], pt_coord: Iterable[Iterable[float]]) -> np.ndarray:
    ref_lat, ref_lng = ref_coord
    pt_coord = pt_coord if type(pt_coord) == np.ndarray else np.array(pt_coord)
    pt_lat, pt_lng = pt_coord[:, 0].ravel(), pt_coord[:, 1].ravel()

    x = R * (pt_lng - ref_lng) * np.cos((ref_lat + pt_lat) / 2)
    y = R * (pt_lat - ref_lat)

    return np.column_stack([x, y])


def latlng_inverse_eq_rect(ref_coord: Iterable[float], pt_coord: Union[Iterable[float], np.ndarray]) -> np.ndarray:
    ref_lat, ref_lng = ref_coord
    pt_coord = pt_coord if type(pt_coord) == np.ndarray else np.array(pt_coord)
    pt_x, pt_y = pt_coord[:, 0].ravel(), pt_coord[:, 1].ravel()

    lat = pt_y / R + ref_lat
    lng = pt_x / (R * np.cos((ref_lat + lat) / 2)) + ref_lng

    return np.column_stack([lat, lng])


class PoiCluster(ClusterMixin):

    def __init__(self, d_max: float, r_max: float, t_max: float, t_min: float):
        """
        Parameters
        ----------
        d_max: float
            Distance threshold in metres for determining two points are close.
        r_max: float
            Width of a stay region in metres
        t_max: float
            Temporal distance threshold for determining two points are too far
        t_min: float
            Temporal distance threshold for determining tow points are close.
        """
        self._d_max = d_max
        self._r_max = r_max
        self._t_max = t_max
        self._t_min = t_min

        self._grid_width = self._r_max / 3

        self._label_keys = []
        self._stay_points = None
        self._stay_time = None
        self._stay_region_grids: Optional[Tuple[float, float], str] = None
        self._ref_coord = None

    @property
    def stay_points_(self, return_stay_time: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """

        Parameters
        ----------
        return_stay_time: bool
            If True, return stay time

        Returns
        -------
        stay_points: np.ndarray, shape = (n_stay_points, 2)
            GPS coordinates in radians (latitude, longitude) of stay points
        stay_time: np.ndarray, shape = (n_stay_points, 2)
            stay time (from, to) of corresponding stay points
        """

        if return_stay_time:
            return self._stay_points, self._stay_time
        else:
            return self._stay_points

    @property
    def stay_region_grids_(self) -> Dict:
        """
        Returns
        -------
        stay_region_grids: dict, key = tuple(from_latitude, from_longitude, to_latitude, to_longitude), value = str
            a key is a rectangular coordinates in radians representing a grid.
            a value is a corresponding cluster id.

        Notes
        ------
        Different grids can have a same cluster id, because those are neighbors.
        """
        r = {}

        for g, c in self._stay_region_grids.items():
            lat0, lng0 = latlng_inverse_eq_rect(self._ref_coord, map(lambda x: x * self._grid_width, g)).ravel()
            lat1, lng1 = latlng_inverse_eq_rect(self._ref_coord, map(lambda x: (x + 1) * self._grid_width, g)).ravel()
            r[tuple([lat0, lng0, lat1, lng1])] = c
        return r

    def fit(self, x: np.ndarray, timestamps: Optional[np.ndarray] = None) -> 'PoiCluster':
        """
        Parameters
        ----------
        x: np.ndarray, shape = (n_samples, 2)
            GPS coordinates (latitude, longitude) in radians.
        timestamps: np.ndarray, optional, shape = (n_samples, )
            arrival time corresponding to each coordinate. If not specified, all coordinates are assumed to be collected
            in same period
        Returns
        -------
        poi_cluster: PoiCluster
        """
        ref_coord = np.min(x, axis=0)
        stay_points, stay_time = self._find_stay_points(x, timestamps)
        stay_regions = self._find_stay_region(ref_coord, stay_points)

        self._ref_coord = ref_coord
        self._stay_points = stay_points
        self._stay_time = stay_time
        self._stay_region_grids = stay_regions

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray, shape = (n_samples, 2)
            GPS coordinates (latitude, longitude) in radians.
        Returns
        -------
        labels: np.ndarray, shape = (n_samples,)
            Cluster labels corresponding to each data sample.
            If the sample does not included in any cluster, it has 'NONE' label.
        """
        assert self._ref_coord is not None, 'You should \'fit\' cluster before \'predict\''
        grids = self._make_grids(self._ref_coord, x)
        labels = [self._stay_region_grids.get(tuple(g)) for g in grids]
        return np.array(labels)

    def fit_predict(self, x: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(x, timestamps).predict(x)

    def _generate_unique_key(self) -> str:
        key = uuid4().hex[:6].upper()
        while key in self._label_keys:
            key = uuid4().hex[:6].upper()
        return key

    def _make_grids(self, ref_coord: np.ndarray, pt_coord: np.ndarray) -> np.ndarray:
        grids = np.floor(latlng_eq_rect(ref_coord, pt_coord) / self._grid_width)
        return grids

    def _find_stay_points(self,
                          coords: np.ndarray,
                          timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        assert coords.shape[1] == 2, '\'coordinates\' should be (n_samples, 2)'
        l, _ = coords.shape
        if timestamps is not None:
            assert len(timestamps.shape) == 1 and timestamps.shape[0] == l, \
                '\'timestamps should have a same length as \'coordinates\''
        else:
            timestamps = np.arange(l)

        stay_points = []
        stay_time = []
        i = 0

        while i < l:
            j = i + 1
            eof = True
            while j < l:
                t_diff = timestamps[j] - timestamps[j - 1]
                if t_diff > self._t_max:
                    i = j
                    eof = False
                    break
                dist = latlng_haversine_dist(coords[i], coords[j])
                if dist > self._d_max:
                    t_diff = timestamps[j - 1] - timestamps[i]
                    if t_diff > self._t_min:
                        lat, lng = latlng_centroid(coords[i:j])
                        stay_points.append((lat, lng))
                        stay_time.append((timestamps[i], timestamps[j - 1]))
                    i = j
                    eof = False
                    break
                j = j + 1
            if eof:
                break
        return np.vstack(stay_points), np.vstack(stay_time)

    def _find_stay_region(self, ref_coord: np.ndarray, stay_points: np.ndarray):
        grids = self._make_grids(ref_coord, stay_points)
        stay_region_ids = np.repeat(NON_LABEL, grids.shape[0]).astype(object)
        stay_region_center = []
        stay_region_grids = {}

        while np.any(stay_region_ids == NON_LABEL):
            g, c = np.unique(grids[stay_region_ids == NON_LABEL], axis=0, return_counts=True)
            g_dense = g[np.argsort(c)[-1]]

            indices = []

            for i, j in product([-1, 0, 1], [-1, 0, 1]):
                ng = g_dense + (i, j)
                # masked array of the neighboring grids within in all grids.
                indices_ng = np.any(np.all(grids[:, None] == ng, axis=2), axis=1)
                # check whether the grid is already assigned to any id.
                if np.any(stay_region_ids[indices_ng] == NON_LABEL):
                    indices.append(np.flatnonzero(indices_ng))

            if len(indices) == 0:
                continue

            indices = np.hstack(indices)
            all_stay_points = stay_points[indices]
            center = tuple(latlng_centroid(all_stay_points))

            if center not in stay_region_center:
                stay_region_center.append(center)
                new_id = self._generate_unique_key()
                stay_region_ids[indices] = new_id
                all_grids = np.unique(grids[indices], axis=0)

                for g in all_grids:
                    stay_region_grids[tuple(g)] = new_id

        return stay_region_grids
