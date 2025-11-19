# limpieza_atlas.py
import pandas as pd
import numpy as np

def limpiar_atlas_pliegues():
    pliegues = pd.read_csv('Data/Originals/Atlas_Geol%C3%B3gico_de_Colombia_2023%3A_Pliegues_2023.csv')
    print(f" Registros originales: {len(pliegues)}")
    print(f" Columnas disponibles: {pliegues.columns.tolist()}")
    
    print("\n Informacion del dataset:")
    print(pliegues.info())
    
    # primeras filas
    print("\n Primeras 5 filas:")
    print(pliegues.head())
    
    # 2. ANALISIS DE CALIDAD
    # valores nulos
    nulos = pliegues.isnull().sum()
    for columna, cantidad in nulos.items():
        print(f"   {columna}: {cantidad} nulos ({cantidad/len(pliegues)*100:.1f}%)")
    
    # duplicados
    duplicados = pliegues.duplicated().sum()
    print(f"\n registros duplicados: {duplicados}")
    
    # 3. ESTADISTICAS POR COLUMNA
    # columnas numericas
    columnas_numericas = pliegues.select_dtypes(include=[np.number]).columns
    if not columnas_numericas.empty:
        print("\n estadisticas numericas:")
        print(pliegues[columnas_numericas].describe())
    
    # columnas categoricas
    columnas_categoricas = pliegues.select_dtypes(include=['object']).columns
    if not columnas_categoricas.empty:
        print("\n columnas categoricas:")
        for col in columnas_categoricas:
            print(f"\n   {col}:")
            print(f"   Valores unicos: {pliegues[col].nunique()}")
            print(f"   Top 5 valores: {pliegues[col].value_counts().head().to_dict()}")
    
    # 4. LIMPIEZA ESPECIFICA PARA DATOS GEOLOGICOS
    pliegues_limpio = pliegues.copy()
    
    # estandarizar textos
    for col in columnas_categoricas:
        if col in pliegues_limpio.columns:
            # eliminar espacios en blanco mayus
            pliegues_limpio[col] = pliegues_limpio[col].astype(str).str.strip().str.title()
            print(f"    {col}: estandarizado")

    pliegues_limpio.to_csv('Data/Procesados/pliegues_limpios.csv', index=False, encoding='utf-8')


if __name__ == "__main__":
    atlas_limpio = limpiar_atlas_pliegues()