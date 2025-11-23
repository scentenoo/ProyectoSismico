import pandas as pd
import numpy as np
import json
from shapely.geometry import Point, shape
from shapely.prepared import prep
import os

def cargar_datos():
    """Cargar todos los datasets"""
    print("📂 Cargando datos...")
    
    pliegues = pd.read_csv('Data/Procesados/pliegues_limpios.csv')
    print(f"   Pliegues: {len(pliegues)} registros")
    
    sismos = pd.read_excel('Data/Procesados/LLCatálogo Sismicidad TECTO_limpio.xlsx')
    print(f"   Sismos: {len(sismos)} registros")
    
    with open('Data/Procesados/MunicipiosColombia.json', 'r', encoding='utf-8') as f:
        municipios_geojson = json.load(f)
    print(f"   Municipios: {len(municipios_geojson['features'])} features")
    
    return pliegues, sismos, municipios_geojson

def corregir_coordenadas(sismos):
    """Corregir las coordenadas del dataset de sismos"""
    print("\n🔧 Corrigiendo coordenadas...")
    
    # Columnas EXACTAS según tu archivo
    lat_col = 'Lat(°)'
    lon_col = 'Long(°)'
    
    print(f"   Valores originales (primeros 3):")
    print(f"   Lat: {sismos[lat_col].head(3).tolist()}")
    print(f"   Lon: {sismos[lon_col].head(3).tolist()}")
    
    # Los valores están multiplicados por 1000
    # 7027 -> 7.027, -74262 -> -74.262
    sismos['lat_corregida'] = sismos[lat_col] / 1000
    sismos['lon_corregida'] = sismos[lon_col] / 1000
    
    print(f"\n   Valores corregidos (primeros 3):")
    print(f"   Lat: {sismos['lat_corregida'].head(3).tolist()}")
    print(f"   Lon: {sismos['lon_corregida'].head(3).tolist()}")
    
    # Verificar rangos
    lat_min, lat_max = sismos['lat_corregida'].min(), sismos['lat_corregida'].max()
    lon_min, lon_max = sismos['lon_corregida'].min(), sismos['lon_corregida'].max()
    
    print(f"\n   Rango Lat: [{lat_min:.3f}, {lat_max:.3f}]")
    print(f"   Rango Lon: [{lon_min:.3f}, {lon_max:.3f}]")
    print(f"   (Colombia: Lat -4 a 13, Lon -82 a -66)")
    
    # Validar
    if -5 <= lat_min and lat_max <= 15 and -85 <= lon_min and lon_max <= -65:
        print("   ✅ Coordenadas válidas para Colombia!")
    else:
        print("   ⚠️ Algunas coordenadas pueden estar fuera de Colombia")
    
    return sismos

def preparar_municipios(municipios_geojson):
    """Preparar geometrías de municipios"""
    print("\n🗺️ Preparando geometrías de municipios...")
    municipios_prep = []
    for feature in municipios_geojson['features']:
        try:
            geom = shape(feature['geometry'])
            municipios_prep.append({
                'geometry': geom,
                'prepared': prep(geom),
                'properties': feature['properties']
            })
        except:
            pass
    print(f"   Municipios preparados: {len(municipios_prep)}")
    return municipios_prep

def asignar_sismos_a_municipios(sismos, municipios_prep):
    """Asignar cada sismo a su municipio"""
    print("\n📍 Asignando sismos a municipios...")
    
    sismos_con_municipio = []
    sismos_sin_municipio = 0
    total = len(sismos)
    
    for idx, sismo in sismos.iterrows():
        if idx % 1000 == 0:
            print(f"   Procesando: {idx}/{total}...")
        
        try:
            lat = sismo['lat_corregida']
            lon = sismo['lon_corregida']
            
            if pd.isna(lat) or pd.isna(lon):
                sismos_sin_municipio += 1
                continue
            
            punto = Point(lon, lat)
            encontrado = False
            
            for mun in municipios_prep:
                if mun['prepared'].contains(punto):
                    props = mun['properties']
                    sismo_dict = sismo.to_dict()
                    sismo_dict.update({
                        'municipio': props.get('NOMBRE_ENT', ''),
                        'departamento': props.get('DEPARTAMEN', ''),
                        'cod_municipio': props.get('COD_MUNICI', ''),
                        'cod_departamento': props.get('COD_DEPART', '')
                    })
                    sismos_con_municipio.append(sismo_dict)
                    encontrado = True
                    break
            
            if not encontrado:
                sismos_sin_municipio += 1
                
        except Exception as e:
            sismos_sin_municipio += 1
    
    print(f"\n   ✅ Sismos asignados: {len(sismos_con_municipio)}")
    print(f"   ❌ Sismos fuera de municipios: {sismos_sin_municipio}")
    
    return pd.DataFrame(sismos_con_municipio)

def crear_dataset_unificado(sismos_con_municipio):
    """Crear dataset agregado por municipio"""
    print("\n🔄 Creando dataset unificado...")
    
    if sismos_con_municipio.empty:
        print("   ❌ No hay sismos asignados")
        return pd.DataFrame()
    
    # Corregir magnitud también (dividir por 1000 si es necesario)
    if 'Mag.' in sismos_con_municipio.columns:
        mag_values = sismos_con_municipio['Mag.']
        if mag_values.max() > 100:  # Magnitudes no pasan de 10
            sismos_con_municipio['Mag.'] = mag_values / 10000
    
    if 'Prof(Km)' in sismos_con_municipio.columns:
        prof_values = sismos_con_municipio['Prof(Km)']
        if prof_values.max() > 1000:  # Profundidades razonables < 700km
            sismos_con_municipio['Prof(Km)'] = prof_values / 1000
    
    sismos_agg = sismos_con_municipio.groupby('cod_municipio').agg({
        'municipio': 'first',
        'departamento': 'first',
        'Mag.': ['count', 'max', 'mean'],
        'Prof(Km)': 'mean',
        'lat_corregida': 'mean',
        'lon_corregida': 'mean'
    }).reset_index()
    
    # Aplanar columnas
    sismos_agg.columns = [
        'cod_municipio', 'municipio', 'departamento',
        'total_sismos', 'magnitud_max', 'magnitud_promedio',
        'profundidad_promedio', 'lat_centroide', 'lon_centroide'
    ]
    
    print(f"   ✅ Dataset creado: {len(sismos_agg)} municipios")
    print(f"\n   📊 Top 5 municipios con más sismos:")
    print(sismos_agg.nlargest(5, 'total_sismos')[['municipio', 'departamento', 'total_sismos']].to_string(index=False))
    
    return sismos_agg

def main():
    print("🚀 INTEGRACIÓN ESPACIAL - SISMOS COLOMBIA")
    print("=" * 50)
    
    # 1. Cargar datos
    pliegues, sismos, municipios_geojson = cargar_datos()
    
    # 2. Corregir coordenadas (dividir por 1000)
    sismos = corregir_coordenadas(sismos)
    
    # 3. Preparar municipios
    municipios_prep = preparar_municipios(municipios_geojson)
    
    # 4. Asignar sismos a municipios
    sismos_con_municipio = asignar_sismos_a_municipios(sismos, municipios_prep)
    
    # 5. Crear dataset unificado
    if not sismos_con_municipio.empty:
        dataset_final = crear_dataset_unificado(sismos_con_municipio)
        
        os.makedirs('Data/Procesados', exist_ok=True)
        
        # Guardar sismos con municipio asignado
        sismos_con_municipio.to_csv('Data/Procesados/sismos_con_municipio.csv', index=False)
        print(f"\n💾 Guardado: sismos_con_municipio.csv")
        
        # Guardar agregado por municipio
        dataset_final.to_csv('Data/Procesados/dataset_unificado.csv', index=False)
        print(f"💾 Guardado: dataset_unificado.csv")
        
        print("\n" + "=" * 50)
        print("🎉 ¡INTEGRACIÓN COMPLETADA!")
    else:
        print("\n❌ No se pudo crear el dataset")

if __name__ == "__main__":
    main()