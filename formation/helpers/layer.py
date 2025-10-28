from typing import Optional

from sqlalchemy import Connection

import pandas as pd

from formation.models import Layer, Point
from sqlalchemy.dialects.sqlite import insert

from formation.models.distance import Distance


def save_layer(
        conn: Connection,
        batch_id: int,
        points: pd.DataFrame,
        distances: pd.DataFrame,
        level: int,
        configuration_id: Optional[int] = None,
):
    layer_id = conn.execute(
        insert(Layer).values(
            batch_id=batch_id,
            configuration_id=configuration_id,
            level=level,
        )
    ).inserted_primary_key[0]

    for _, point in points.iterrows():
        st_point = insert(Point).values(
            layer_id=layer_id,
            code=point['code'],
            lat=point['lat'],
            lng=point['lng'],
            x=point['x'],
            y=point['y'],
            cluster=point['cluster'] if 'cluster' in point else None,
        )
        conn.execute(st_point)

    for _, point in distances.iterrows():
        st_distance = insert(Distance).values(
            lat_in=str(point['lat_in']),
            lng_in=str(point['lng_in']),
            lat_out=str(point['lat_out']),
            lng_out=str(point['lng_out']),
            distance=point['distance'],
        )
        conn.execute(st_distance.on_conflict_do_nothing(index_elements=['lat_in', 'lng_in', 'lat_out', 'lng_out']))