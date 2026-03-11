from __future__ import annotations
import argparse
import heapq
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterio import features
from rasterio.transform import from_origin, xy as rio_xy
import matplotlib.pyplot as plt

# We try to import medial_axis first and fall back to skeletonize if it's not available.
try:
    from skimage.morphology import medial_axis
    USE_MEDIAL_AXIS = True
except Exception:
    from skimage.morphology import skeletonize
    USE_MEDIAL_AXIS = False

# Geometry configurations
def make_valid_geom(geom):
    try:
        from shapely.validation import make_valid as _mv
        return _mv(geom)
    except Exception:
        return geom.buffer(0)

def dissolve_biggest_polygon(gdf: gpd.GeoDataFrame):
    g = unary_union(gdf.geometry)
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda p: p.area)
    return g


def remove_holes(geom):
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    if isinstance(geom, MultiPolygon):
        parts = [Polygon(p.exterior) for p in geom.geoms]
        return unary_union(parts)
    return geom

# Raster  
def rasterize_polygon(poly, res_m: float, pad_cells: int = 2):
    minx, miny, maxx, maxy = poly.bounds
    pad = res_m * pad_cells
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    width = int(math.ceil((maxx - minx) / res_m))
    height = int(math.ceil((maxy - miny) / res_m))
    transform = from_origin(minx, maxy, res_m, res_m)

    arr = features.rasterize(
        [(poly, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return arr, transform


def idx_to_xy(r: int, c: int, transform):
    x, y = rio_xy(transform, r, c, offset="center")
    return float(x), float(y)

# Skeleton graph 

def build_adj_from_skeleton(skel: np.ndarray):
    H, W = skel.shape
    nodes: List[Tuple[int, int]] = []
    idx_of = {}

    for r in range(H):
        for c in range(W):
            if skel[r, c]:
                idx_of[(r, c)] = len(nodes)
                nodes.append((r, c))

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(len(nodes))]
    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]
    for (r, c), i in idx_of.items():
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            j = idx_of.get((rr, cc))
            if j is not None:
                w = math.sqrt(2) if (dr and dc) else 1.0
                adj[i].append((j, w))
    return nodes, adj


def endpoints_from_adj(adj):
    return [i for i, nbrs in enumerate(adj) if len(nbrs) == 1]


def dijkstra_unit(adj, src: int):
    n = len(adj)
    dist = [math.inf] * n
    prev = [-1] * n
    dist[src] = 0.0
    pq = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for (v, _w) in adj[u]:
            nd = d + 1.0
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev, dst: int):
    path = []
    u = dst
    while u != -1:
        path.append(u)
        u = prev[u]
    return list(reversed(path))


def endpoints_diameter(adj) -> Tuple[Optional[List[int]], str]:
    eps = endpoints_from_adj(adj)
    if len(eps) < 2:
        return None, "endpoints<2"

    A = eps[0]
    distA, _ = dijkstra_unit(adj, A)
    B = max(eps, key=lambda i: distA[i] if np.isfinite(distA[i]) else -1)
    if not np.isfinite(distA[B]):
        return None, "disconnected_A"

    distB, prevB = dijkstra_unit(adj, B)
    C = max(eps, key=lambda i: distB[i] if np.isfinite(distB[i]) else -1)
    if not np.isfinite(distB[C]):
        return None, "disconnected_B"

    return reconstruct_path(prevB, C), "ok"


def boundary_nodes(poly, nodes, transform, tol_m: float):
    idxs = []
    boundary = poly.boundary
    for i, (r, c) in enumerate(nodes):
        x, y = idx_to_xy(r, c, transform)
        if boundary.distance(Point(x, y)) <= tol_m:
            idxs.append(i)
    return idxs


def subset_diameter(adj, subset_idxs: List[int]) -> Optional[List[int]]:
    if len(subset_idxs) < 2:
        return None
    A = subset_idxs[0]
    distA, _ = dijkstra_unit(adj, A)
    B = max(subset_idxs, key=lambda i: distA[i] if np.isfinite(distA[i]) else -1)
    if not np.isfinite(distA[B]):
        return None
    distB, prevB = dijkstra_unit(adj, B)
    C = max(subset_idxs, key=lambda i: distB[i] if np.isfinite(distB[i]) else -1)
    if not np.isfinite(distB[C]):
        return None
    return reconstruct_path(prevB, C)


def prune_leaves(nodes, adj, res_m: float, prune_len_m: float):
    alive = np.ones(len(adj), dtype=bool)
    changed = True

    while changed:
        changed = False
        leaves = [
            i for i in range(len(adj))
            if alive[i] and sum(alive[j] for j, _ in adj[i]) <= 1
        ]

        for leaf in leaves:
            if not alive[leaf]:
                continue

            length_edges = 0
            prev_node = -1
            cur = leaf
            ok = True

            while ok:
                nbrs = [v for (v, _w) in adj[cur] if alive[v] and v != prev_node]
                deg_now = len(nbrs) + (1 if prev_node != -1 else 0)
                if deg_now != 2:
                    break
                nxt = nbrs[0]
                length_edges += 1
                if length_edges * res_m >= prune_len_m:
                    ok = False
                    break
                prev_node, cur = cur, nxt

            if length_edges * res_m < prune_len_m:
                alive[leaf] = False
                changed = True

    # compact graph
    index_map = -np.ones(len(adj), dtype=int)
    nodes2 = []
    for i, ok in enumerate(alive):
        if ok:
            index_map[i] = len(nodes2)
            nodes2.append(nodes[i])

    adj2 = [[] for _ in range(len(nodes2))]
    for i, ok in enumerate(alive):
        if not ok:
            continue
        ii = index_map[i]
        for (j, w) in adj[i]:
            if alive[j]:
                jj = index_map[j]
                adj2[ii].append((jj, w))

    return nodes2, adj2



# Line post-processing

def chaikin_smooth(coords: List[Tuple[float, float]], iterations: int = 1):
    for _ in range(iterations):
        if len(coords) < 2:
            return coords
        new = [coords[0]]
        for i in range(len(coords) - 1):
            p0 = np.array(coords[i])
            p1 = np.array(coords[i + 1])
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new.extend([tuple(Q), tuple(R)])
        new.append(coords[-1])
        coords = new
    return coords


def densify_linestring(line: LineString, spacing_m: float):
    if line.length <= spacing_m:
        return line
    n = int(math.ceil(line.length / spacing_m))
    dists = np.linspace(0, line.length, n + 1)
    pts = [line.interpolate(d) for d in dists]
    return LineString([(p.x, p.y) for p in pts])


def coords_from_nodes(nodes, idxs, transform):
    return [idx_to_xy(nodes[i][0], nodes[i][1], transform) for i in idxs]



# Pipeline

@dataclass
class Config:
    in_path: str
    layer: Optional[str]
    out_gpkg: str
    out_layer: str
    out_geojson: str
    fig_png: Optional[str]
    fig_svg: Optional[str]
    epsg: int
    res_m: float
    pad_cells: int
    pre_buffer_m: float
    remove_holes_flag: bool
    prune_len_m: float
    bound_tol_m: float
    smooth_iter: int
    densify_m: float

# Main extraction function, by steps
def extract_thalweg(cfg: Config) -> LineString:
    # 1) Read polygon
    if not os.path.exists(cfg.in_path):
        raise FileNotFoundError(f"Input file not found: {cfg.in_path}")

    try:
        if cfg.layer:
            gdf = gpd.read_file(cfg.in_path, layer=cfg.layer)
        else:
            gdf = gpd.read_file(cfg.in_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed reading input '{cfg.in_path}'"
            + (f" (layer '{cfg.layer}')" if cfg.layer else "")
            + f". Error: {e}"
        )

    if len(gdf) == 0:
        raise RuntimeError("Input polygon layer is empty (0 features).")

    gdf = gdf.to_crs(epsg=cfg.epsg)

    poly = dissolve_biggest_polygon(gdf)
    poly = make_valid_geom(poly)

    if cfg.remove_holes_flag:
        poly = remove_holes(poly)

    if cfg.pre_buffer_m and cfg.pre_buffer_m > 0:
        poly = poly.buffer(cfg.pre_buffer_m)

    # 2) Rasterize & skeletonize
    mask, transform = rasterize_polygon(poly, cfg.res_m, pad_cells=cfg.pad_cells)
    binary = mask.astype(bool)

    if USE_MEDIAL_AXIS:
        skel_bool, _ = medial_axis(binary, return_distance=True)
    else:
        from skimage.morphology import skeletonize as _sk
        skel_bool = _sk(binary)

    skel = skel_bool.astype(np.uint8)
    if int(skel.sum()) == 0:
        raise RuntimeError("Skeleton is empty. Try coarser --res or check polygon geometry.")

    # 3) Graph
    nodes, adj = build_adj_from_skeleton(skel)
    if not nodes:
        raise RuntimeError("No skeleton nodes. Check mask/skeletonization.")

    # 4) A) Diameter after pruning (preferred)
    nodes_p, adj_p = prune_leaves(nodes, adj, cfg.res_m, cfg.prune_len_m)
    path_idx, status = endpoints_diameter(adj_p)

    # 5) B) Fallback: diameter without pruning
    if path_idx is None:
        path_idx, status = endpoints_diameter(adj)

    # 6) C) Fallback: loop case -> boundary subset diameter
    if path_idx is None:
        b_idxs = boundary_nodes(poly, nodes, transform, cfg.bound_tol_m)
        path_idx = subset_diameter(adj, b_idxs)
        if path_idx is not None:
            status = "boundary_subset"

    # 7) D) Ultimate fallback: two-sweep over all nodes
    if path_idx is None:
        dist0, _ = dijkstra_unit(adj, 0)
        B = int(np.nanargmax([d if np.isfinite(d) else -1 for d in dist0]))
        distB, prevB = dijkstra_unit(adj, B)
        C = int(np.nanargmax([d if np.isfinite(d) else -1 for d in distB]))
        path_idx = reconstruct_path(prevB, C)
        status = "all_nodes_fallback"

    coords = coords_from_nodes(nodes, path_idx, transform)
    if len(coords) < 2:
        raise RuntimeError(f"Degenerate path (len<2). status={status}")

    line = LineString(coords)

    # 8) Smooth + densify
    if cfg.smooth_iter > 0:
        line = LineString(chaikin_smooth(list(line.coords), cfg.smooth_iter))
    if cfg.densify_m and cfg.densify_m > 0:
        line = densify_linestring(line, cfg.densify_m)

    # 9) Save outputs
    out = gpd.GeoDataFrame({"name": [cfg.out_layer]}, geometry=[line], crs=f"EPSG:{cfg.epsg}")
    os.makedirs(os.path.dirname(cfg.out_gpkg) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.out_geojson) or ".", exist_ok=True)

    out.to_file(cfg.out_gpkg, layer=cfg.out_layer, driver="GPKG")
    out.to_file(cfg.out_geojson, driver="GeoJSON")

def parse_args():

    p = argparse.ArgumentParser(
        description="Extract thalweg from a polygon using skeletonization."
    )
    p.add_argument("--print_versions", action="store_true", help="Print software versions and exit.")

    # Inputs/outputs (validated in main() unless --print_versions)
    p.add_argument("--in", dest="in_path", default=None, help="Input polygon file (gpkg/shp/etc.).")
    p.add_argument("--layer", default=None, help="Layer name if input is a GeoPackage.")
    p.add_argument("--out_gpkg", default=None, help="Output GeoPackage path.")
    p.add_argument("--out_layer", default="thalweg", help="Output layer name in GPKG.")
    p.add_argument("--out_geojson", default=None, help="Output GeoJSON path.")
    p.add_argument("--fig_png", default=None, help="Optional output PNG figure path.")
    p.add_argument("--fig_svg", default=None, help="Optional output SVG figure path.")

    # Parameters
    p.add_argument("--epsg", type=int, default=32628, help="Target EPSG (default 32628).")
    p.add_argument("--res", dest="res_m", type=float, default=25.0, help="Raster resolution (m), default 25.")
    p.add_argument("--pad_cells", type=int, default=2, help="Padding in raster cells, default 2.")
    p.add_argument("--pre_buffer", dest="pre_buffer_m", type=float, default=0.0, help="Optional buffer (m) on polygon.")
    p.add_argument("--remove_holes", action="store_true",
                   help="Remove holes by keeping exterior rings only.")
    p.add_argument("--prune_len", dest="prune_len_m", type=float, default=500.0,
                   help="Prune leaf branches shorter than this (m), default 500.")
    p.add_argument("--bound_tol", dest="bound_tol_m", type=float, default=None,
                   help="Boundary tolerance (m). Default = 2*res.")
    p.add_argument("--smooth_iter", type=int, default=1, help="Chaikin smoothing iterations (default 1).")
    p.add_argument("--densify", dest="densify_m", type=float, default=150.0, help="Densify spacing (m), default 150.")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate required args when not printing versions
    required = {
        "--in": args.in_path,
        "--out_gpkg": args.out_gpkg,
        "--out_geojson": args.out_geojson,
    }
    missing = [k for k, v in required.items() if v in (None, "")]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    bound_tol = args.bound_tol_m if args.bound_tol_m is not None else 2.0 * args.res_m

    cfg = Config(
        in_path=args.in_path,
        layer=args.layer,
        out_gpkg=args.out_gpkg,
        out_layer=args.out_layer,
        out_geojson=args.out_geojson,
        fig_png=args.fig_png,
        fig_svg=args.fig_svg,
        epsg=args.epsg,
        res_m=args.res_m,
        pad_cells=args.pad_cells,
        pre_buffer_m=args.pre_buffer_m,
        remove_holes_flag=args.remove_holes,
        prune_len_m=args.prune_len_m,
        bound_tol_m=bound_tol,
        smooth_iter=args.smooth_iter,
        densify_m=args.densify_m,
    )

    line = extract_thalweg(cfg)

    print("Done.")
    print(f"Length (m): {line.length:.1f} | vertices: {len(line.coords)}")
    print(f"Saved: {cfg.out_gpkg} (layer '{cfg.out_layer}')")
    print(f"Saved: {cfg.out_geojson}")
    if cfg.fig_png:
        print(f"Saved: {cfg.fig_png}")
    if cfg.fig_svg:
        print(f"Saved: {cfg.fig_svg}")


if __name__ == "__main__":
    main()
