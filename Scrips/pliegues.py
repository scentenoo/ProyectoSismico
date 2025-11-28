import pandas as pd
import numpy as np
import os

print("="*80)
print("LIMPIEZA - PLIEGUES GEOLÓGICOS")
print("="*80)

# Rutas
archivo_entrada = r'Data\Originals\Atlas_Geol%C3%B3gico_de_Colombia_2023%3A_Pliegues_2023.csv'
carpeta_salida = 'Data/Procesados'
archivo_salida = os.path.join(carpeta_salida, 'pliegues_limpios.csv')

os.makedirs(carpeta_salida, exist_ok=True)

# Cargar datos con encoding correcto
print("\n[1/5] Cargando archivo...")
df = pd.read_csv(archivo_entrada, encoding='utf-8-sig')  # Corrige el BOM
print(f"Registros: {len(df)}")
print(f"Columnas: {len(df.columns)}")

# Ver estructura
print("\nColumnas:")
for i, col in enumerate(df.columns, 1):
    nulos = df[col].isnull().sum()
    unicos = df[col].nunique()
    print(f"  {i}. {col:20s} - Nulos: {nulos:4d} - Únicos: {unicos:4d}")

# Trabajar con copia
df_limpio = df.copy()

# Verificar nulos
print("\n[2/5] Verificando nulos...")
nulos_total = df_limpio.isnull().sum().sum()
if nulos_total > 0:
    print(f"Nulos encontrados: {nulos_total}")
    df_limpio = df_limpio.dropna()
    print(f"Registros después: {len(df_limpio)}")
else:
    print("Sin nulos")

# Duplicados
print("\n[3/5] Verificando duplicados...")
dups = df_limpio.duplicated().sum()
if dups > 0:
    print(f"Duplicados: {dups}")
    df_limpio = df_limpio.drop_duplicates()
else:
    print("Sin duplicados")

# Limpiar texto
print("\n[4/5] Limpiando texto...")
for col in df_limpio.select_dtypes(include=['object']).columns:
    df_limpio[col] = df_limpio[col].str.strip()

print("Texto limpio")

# Análisis
print("\n[5/5] Analizando datos...")

if 'Tipo' in df_limpio.columns:
    print("\nTipos de pliegues:")
    tipos = df_limpio['Tipo'].value_counts()
    for tipo, count in tipos.head(5).items():
        pct = (count / len(df_limpio)) * 100
        print(f"  {tipo}: {count} ({pct:.1f}%)")
    
    if len(tipos) > 5:
        print(f"  ... y {len(tipos) - 5} tipos más")

if 'NombrePlie' in df_limpio.columns:
    nombres_unicos = df_limpio['NombrePlie'].nunique()
    print(f"\nPliegues nombrados: {nombres_unicos}")
    
    # Ejemplos corregidos
    ejemplos = df_limpio['NombrePlie'].dropna().head(5)
    print("Ejemplos:")
    for nombre in ejemplos:
        if nombre.strip():
            print(f"  - {nombre}")

# Guardar
print("\nGuardando...")
df_limpio.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
print(f"CSV: {archivo_salida}")

# Resumen
print("\n" + "="*80)
print("RESUMEN")
print("="*80)
print(f"Registros: {len(df)} → {len(df_limpio)}")
print(f"Tipos de pliegues: {df_limpio['Tipo'].nunique()}")
print(f"Más común: {df_limpio['Tipo'].value_counts().index[0]}")
print(f"Calidad: OK")
print("="*80)