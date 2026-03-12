import numpy as np
import geopandas as gpd
import shapely
from shapely.ops import transform
import pyproj
from pathlib import Path
import urllib.request

_NE_COUNTRIES_110M_ZIP_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

def _load_world():
    """
    Load a world countries GeoDataFrame.

    GeoPandas removed the built-in `gpd.datasets` in v1.0, so we fetch Natural Earth
    data directly and cache it under `data/naturalearth/`.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "naturalearth"
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "ne_110m_admin_0_countries.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(_NE_COUNTRIES_110M_ZIP_URL, zip_path)  # noqa: S310

    return gpd.read_file(zip_path)

world = _load_world()
land = world.geometry.union_all()

def _guess_country_name_column(gdf: gpd.GeoDataFrame) -> str:
    """
    Natural Earth schemas vary by version; prefer common name-like columns.
    """
    candidates = [
        "name",
        "NAME",
        "admin",
        "ADMIN",
        "name_long",
        "NAME_LONG",
        "geounit",
        "GEOUNIT",
        "sovereignt",
        "SOVEREIGNT",
        "formal_en",
        "FORMAL_EN",
    ]

    cols = list(gdf.columns)
    lower_to_actual = {c.lower(): c for c in cols}

    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]

    # fallback: first string/object column that isn't geometry
    for c in cols:
        if c == "geometry":
            continue
        if gdf[c].dtype == "object":
            return c

    raise ValueError("Could not determine a country name column in Natural Earth dataset.")

def get_country_geometry(country_name: str):
    name_col = _guess_country_name_column(world)
    # Case-insensitive match to be user-friendly in Streamlit
    series = world[name_col].astype(str)
    country = world[series.str.casefold() == str(country_name).casefold()]
    if country.empty:
        raise ValueError(f"Country not found: {country_name} (searched column '{name_col}')")
    return country.geometry.union_all()

def local_equal_area_crs(geom):
    lon, lat = geom.centroid.x, geom.centroid.y
    proj4 = (
        f"+proj=laea +lat_0={lat} +lon_0={lon} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return pyproj.CRS.from_proj4(proj4)

def random_points_in_country_equal_area(country_name, n, batch_size=50000, seed=None):
    rng = np.random.default_rng(seed)

    geom_wgs84 = get_country_geometry(country_name)
    crs_proj = local_equal_area_crs(geom_wgs84)

    to_proj = pyproj.Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True).transform

    geom_proj = transform(to_proj, geom_wgs84)

    minx, miny, maxx, maxy = geom_proj.bounds

    xs_out = []
    ys_out = []
    total = 0

    while total < n:
        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)

        mask = shapely.contains_xy(geom_proj, xs, ys)
        if np.any(mask):
            accepted_x = xs[mask]
            accepted_y = ys[mask]

            need = n - total
            accepted_x = accepted_x[:need]
            accepted_y = accepted_y[:need]

            xs_out.append(accepted_x)
            ys_out.append(accepted_y)
            total += len(accepted_x)

    x = np.concatenate(xs_out)
    y = np.concatenate(ys_out)

    # Transform arrays back to WGS84 directly (avoids shapely.ops.transform on ndarray)
    lon, lat = pyproj.Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True).transform(x, y)
    return np.column_stack([lon, lat])

def random_points_on_land_fast(n, batch_size=200000, seed=None):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = land.bounds

    xs_out = []
    ys_out = []
    total = 0

    while total < n:
        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)

        mask = shapely.contains_xy(land, xs, ys)
        if np.any(mask):
            accepted_x = xs[mask]
            accepted_y = ys[mask]

            need = n - total
            accepted_x = accepted_x[:need]
            accepted_y = accepted_y[:need]

            xs_out.append(accepted_x)
            ys_out.append(accepted_y)
            total += len(accepted_x)

    return np.column_stack([np.concatenate(xs_out), np.concatenate(ys_out)])

if __name__ == "__main__":
    pts = random_points_on_land_fast(10000, seed=42)
    print(pts[:5])

    # points = random_points_in_country_equal_area("Germany", 20000, seed=42)
    # print(points[:5])
    # print(len(points))