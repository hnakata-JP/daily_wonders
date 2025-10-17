# Please install apps below first
# pip install "osmnx>=1.9" "geopandas>=0.14" networkx folium shapely scipy

import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import folium

# lat, lon = 35.713756567939065, 139.7634112952661
# G = ox.graph_from_point((lat, lon), dist=4000, network_type="walk")
# ox.save_graphml(G, "utokyo_4km_walk.graphml")
# print("utokyo_4km_walk.graphml を保存しました。")

def down_load_and_save_graph(
    location, dist=4000, network_type="walk", filename="test.graphml"):
    lat, lon = location
    G = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type)
    ox.save_graphml(G, filename)
    print(f"{filename} is saved.")

# e.g., G = ox.load_graphml("utokyo_4km_walk.graphml")
def get_origin_node(G, lat, lon):
    G_proj = ox.project_graph(G)  # メートル系
    pt_wgs84 = gpd.GeoSeries([Point(lon, lat)], crs=G.graph.get("crs", "EPSG:4326"))
    pt_proj  = pt_wgs84.to_crs(G_proj.graph["crs"])
    x, y = pt_proj.iloc[0].x, pt_proj.iloc[0].y
    origin = ox.distance.nearest_nodes(G_proj, X=x, Y=y)
    return origin, G_proj

def iso_polygon_from_nodes(G_proj, reachable_nodes: set, buffer_m=10, simplify_m=3):
    """Detect reachable edges from reachable nodes, then buffer, union_all, and return Polygon (WGS84)"""
    if not reachable_nodes:
        return None

    # employ (u,v,key) MultiIndex to filter edges whose both endpoints are in reachable_nodes
    def both_endpoints_reached(idx):
        try:
            u, v = idx[0], idx[1]
        except Exception:
            return False
        return (u in reachable_nodes) and (v in reachable_nodes)

    edges_gdf_proj = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)
    sub = edges_gdf_proj.loc[edges_gdf_proj.index.map(both_endpoints_reached)]
    if sub.empty:
        return None

    # Buffer and merge road geometries
    merged = sub.buffer(buffer_m).union_all().buffer(0).simplify(simplify_m)

    # Convert to WGS84 and return shapely geometry
    return gpd.GeoDataFrame(geometry=[merged], crs=edges_gdf_proj.crs).to_crs(4326).iloc[0].geometry

def get_base_map(location, zoom_start=16, tiles="cartodbpositron"):
    lat, lon = location
    m = folium.Map(location=[lat, lon], zoom_start=zoom_start, tiles=tiles)
    return m

def add_poly(m, geom, name, color, fill_opacity=0.28):
    if geom is None:
        return
    folium.GeoJson(
        data=gpd.GeoSeries([geom], crs="EPSG:4326").to_json(),
        name=name,
        style_function=lambda x, col=color: {
            "color": col, "weight": 2, "fillOpacity": fill_opacity, "fillColor": col
        },
        tooltip=name,
    ).add_to(m)
