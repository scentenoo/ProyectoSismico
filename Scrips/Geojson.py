import pandas as pd
import json

ruta_json = "Data/Procesados/MunicipiosColombia.json"
with open(ruta_json, 'r', encoding='utf-8') as f:
    municipios_geojson = json.load(f)

municipios_df = pd.DataFrame([feature['properties'] for feature in municipios_geojson['features']])

print("🏙️  Municipios cargados:")
print(f"Número de municipios: {len(municipios_df)}")
print(f"Columnas: {municipios_df.columns.tolist()}")
print("\nPrimeros municipios:")
print(municipios_df.head())