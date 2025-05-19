# SPDX-License-Identifier: MIT
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from difflib import SequenceMatcher


class ArchiveGeolocator:
    """
    Geocoding from a CSV or GeoDataFrame to a reference GeoJSON, only re-geocoding entities that haven't been located yet
    """
    def __init__(self, reference_geojson_path, default_point=Point(0, 0)):
        self.reference = gpd.read_file(reference_geojson_path)
        self.default_point = default_point

    def _combine_fields(self, row, fields):
        return " ".join(
            str(row[f]).strip().lower()
            for f in fields
            if f in row and pd.notna(row[f])
        )

    def _extract_point(self, geom):
        if isinstance(geom, Point):
            return geom
        if isinstance(geom, (LineString, Polygon)):
            return geom.centroid
        return None

    def geocode_csv(
        self,
        csv_path: str = None,
        gdf: gpd.GeoDataFrame = None,
        csv_fields: list[str] = ["Adresse"],
        ref_fields: list[str] = ["nom_voie"],
        output_geometry: str = "geometry",
        match_label: str = "matched_ref",
        score_label: str = "match_score",
        similarity_threshold: float = 80,
        sep: str = ";"
        ) -> gpd.GeoDataFrame:
        """
        If 'gdf' is provided, it will be used; otherwise, 'csv_path' will be loaded.
        Only geocodes rows where geometry is None or equals default_point.
        Returns a complete GeoDataFrame with:
         - the 'output_geometry' column filled,
         - 'match_label' and 'score_label' columns added.
        """
        # Load or reuse the GeoDataFrame
        if gdf is None:
            df = pd.read_csv(csv_path, sep=sep)
            gdf = gpd.GeoDataFrame(df.copy(),
                                   geometry=[None]*len(df),
                                   crs=self.reference.crs)
        else:
            gdf = gdf.copy()
            gdf.set_crs(self.reference.crs, inplace=True, allow_override=True)

        # Loop through and only geocode entities that are not yet located
        for idx, row in gdf.iterrows():
            geom = row.get(output_geometry)
            if geom is not None and geom != self.default_point:
                continue

            query = self._combine_fields(row, csv_fields)
            best_score, best_match, best_geom = 0, None, None

            for _, ref_row in self.reference.iterrows():
                candidate = self._combine_fields(ref_row, ref_fields)
                score = SequenceMatcher(None, query, candidate).ratio() * 100
                if score > best_score:
                    best_score, best_match = score, candidate
                    best_geom = self._extract_point(ref_row.geometry)

            if best_score >= similarity_threshold and best_geom is not None:
                gdf.at[idx, output_geometry] = best_geom
                gdf.at[idx, match_label] = best_match
                gdf.at[idx, score_label] = round(best_score, 1)
            else:
                # no match → use default geometry
                gdf.at[idx, output_geometry] = self.default_point
                gdf.at[idx, match_label] = None
                gdf.at[idx, score_label] = None

        return gdf