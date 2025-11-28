import pandas as pd
import numpy as np
import os

archivo_entrada = r'Data\Originals\Atlas_Geol%C3%B3gico_de_Colombia_2023%3A_Pliegues_2023.csv'
carpeta_salida = 'Data/Procesados'
archivo_salida = os.path.join(carpeta_salida, 'pliegues_limpios.csv')

os.makedirs(carpeta_salida, exist_ok=True)
df = pd.read_csv(archivo_entrada, encoding='utf-8-sig')
print(f"Registros: {len(df)}")
print(f"Columnas: {len(df.columns)}")

for i, col in enumerate(df.columns, 1):
    nulos = df[col].isnull().sum()
    unicos = df[col].nunique()
    print(f"  {i}. {col:20s} - Nulos: {nulos:4d} - Únicos: {unicos:4d}")

df_limpio = df.copy()

nulos_total = df_limpio.isnull().sum().sum()
if nulos_total > 0:
    print(f"Nulos encontrados: {nulos_total}")
    df_limpio = df_limpio.dropna()
    print(f"Registros despues: {len(df_limpio)}")
else:
    print("Sin nulos")

dups = df_limpio.duplicated().sum()
if dups > 0:
    print(f"Duplicados: {dups}")
    df_limpio = df_limpio.drop_duplicates()
else:
    print("Sin duplicados")

for col in df_limpio.select_dtypes(include=['object']).columns:
    df_limpio[col] = df_limpio[col].str.strip()

df_limpio.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
print(f"CSV: {archivo_salida}")