# SPDX-License-Identifier: MIT
class GeoExporter:
    @staticmethod
    def to_geojson(gdf, path):
        gdf.to_file(path, driver="GeoJSON")

    @staticmethod
    def to_csv(gdf, path, sep=';'):
        geom_col = gdf.geometry.name
        df = gdf.drop(columns=[geom_col])
        df.to_csv(path, index=False, sep=sep)