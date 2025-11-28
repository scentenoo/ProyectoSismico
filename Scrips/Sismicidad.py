import pandas as pd
import numpy as np
import os
from pathlib import Path

RUTA_ENTRADA = 'Data/Originals/Catálogo Sismicidad TECTO.csv'
CARPETA_SALIDA = 'Data/Procesados'
NOMBRE_SALIDA = 'LLCatálogo Sismicidad TECTO_limpio.xlsx'

RANGOS_VALIDACION = {
    'magnitud': (0, 10),
    'latitud': (-5, 15),
    'longitud': (-85, -60),
    'profundidad': (0, 700)
}

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df

def identificar_columnas(df):
    mapeo = {}
    patrones = {
        'fecha_hora': ['fecha', 'hora'],
        'latitud': ['lat'],
        'longitud': ['long'],
        'profundidad': ['prof'],
        'magnitud': ['mag'],
        'region': ['region', 'municipio'],
        'rms': ['rms'],
        'gap': ['gap']
    }
    
    for clave, palabras in patrones.items():
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in palabras):
                if 'error' not in col_lower and ('tipo' not in col_lower or clave != 'magnitud'):
                    mapeo[clave] = col
                    break
    
    print("\n📊 Columnas identificadas:")
    for k, v in mapeo.items():
        print(f"   • {k}: '{v}'")
    
    return mapeo

def limpiar_nulos(df, cols_criticas):
    inicial = len(df)
    df_limpio = df.dropna(subset=cols_criticas)
    eliminados = inicial - len(df_limpio)
    
    if eliminados > 0:
        print(f"\n🗑️  Eliminados {eliminados} registros con datos faltantes")
    
    return df_limpio

def convertir_tipos(df, cols_numericas):
    """Convierte columnas a tipos numéricos"""
    for col in cols_numericas:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_limpio = df.dropna(subset=[c for c in cols_numericas if c])
    
    return df_limpio

def validar_rangos(df, mapeo):
    eliminados_total = 0
    
    for campo, (min_val, max_val) in RANGOS_VALIDACION.items():
        if campo in mapeo:
            col = mapeo[campo]
            invalidos = (df[col] < min_val) | (df[col] > max_val)
            n_invalidos = invalidos.sum()
            
            if n_invalidos > 0:
                df = df[~invalidos]
                print(f"   • {campo}: {n_invalidos} valores fuera de rango")
                eliminados_total += n_invalidos
    
    return df

def limpiar_texto(df, col_region):
    if col_region in df.columns:
        df[col_region] = df[col_region].str.strip()
        df = df[df[col_region] != '']
    
    return df
def main():
    Path(CARPETA_SALIDA).mkdir(parents=True, exist_ok=True)
    
    # Pipeline de limpieza
    print("\n[1/6] Cargando datos...")
    df_original = cargar_datos(RUTA_ENTRADA)
    
    print("\n[2/6] Identificando columnas...")
    mapeo = identificar_columnas(df_original)
    
    print("\n[3/6] Limpiando datos nulos...")
    cols_criticas = [mapeo[k] for k in ['magnitud', 'latitud', 'longitud', 'profundidad', 'region'] if k in mapeo]
    df = limpiar_nulos(df_original, cols_criticas)
    
    print("\n[4/6] Convirtiendo tipos de datos...")
    cols_numericas = [mapeo[k] for k in ['magnitud', 'latitud', 'longitud', 'profundidad', 'rms', 'gap'] if k in mapeo]
    df = convertir_tipos(df, cols_numericas)
    
    print("\n[5/6] Validando rangos...")
    df = validar_rangos(df, mapeo)
    
    if 'region' in mapeo:
        df = limpiar_texto(df, mapeo['region'])
    
    # Eliminar duplicados
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        df = df.drop_duplicates()
        print(f"\n🔄 Eliminados {duplicados} registros duplicados")
    
    print("\n[6/6] Guardando archivo limpio...")
    ruta_salida = os.path.join(CARPETA_SALIDA, NOMBRE_SALIDA)
    df.to_excel(ruta_salida, index=False)
    print(f"✓ Guardado en: {ruta_salida}")
    
    return df

if __name__ == "__main__":
    df_limpio = main()