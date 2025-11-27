import pandas as pd
import os

# ==================== CONFIGURACIÓN ====================
# Nombre de tu archivo CSV
archivo = 'Data/Procesados/pliegues_limpios.csv'

# Carpeta donde se guardarán los resultados
carpeta_resultados = 'Data/Procesados'

print("="*60)
print("🗻 ANÁLISIS DE DATOS DE PLIEGUES")
print("="*60)

# ==================== CARGAR DATOS ====================
print(f"\n📥 Cargando datos desde {archivo}...")
try:
    df = pd.read_csv(archivo)
    print(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas\n")
except Exception as e:
    print(f"❌ Error al cargar datos: {e}")
    print("Verifica que el archivo CSV esté en la ruta correcta.")
    exit()

# ==================== MOSTRAR COLUMNAS ====================
print("📋 Columnas disponibles:")
print(df.columns.tolist())
print()

# ==================== IDENTIFICAR COLUMNAS CLAVE ====================
# Buscar columnas de longitud
columna_longitud = None
posibles_longitud = ['longitud', 'Longitud', 'LENGTH', 'Length', 'SHAPE_Length', 'Shape_Length']

for nombre in posibles_longitud:
    if nombre in df.columns:
        columna_longitud = nombre
        break

# Buscar columnas de tipo de pliegue
columna_tipo = None
posibles_tipo = ['tipo', 'Tipo', 'Type', 'tipo_pliegue', 'Tipo_Pliegue', 'TIPO']

for nombre in posibles_tipo:
    if nombre in df.columns:
        columna_tipo = nombre
        break

# Buscar columnas de área o coordenadas para densidad
columna_area = None
posibles_area = ['area', 'Area', 'AREA', 'superficie', 'Superficie']

for nombre in posibles_area:
    if nombre in df.columns:
        columna_area = nombre
        break

print(f"✅ Columnas identificadas:")
print(f"   Longitud: {columna_longitud if columna_longitud else 'No encontrada'}")
print(f"   Tipo: {columna_tipo if columna_tipo else 'No encontrada'}")
print(f"   Área: {columna_area if columna_area else 'No encontrada'}\n")

# ==================== LIMPIAR DATOS ====================
print("🧹 Limpiando datos...")

# Limpiar longitud si existe
if columna_longitud:
    df[columna_longitud] = pd.to_numeric(df[columna_longitud], errors='coerce')
    df_limpio = df.dropna(subset=[columna_longitud])
    print(f"   Registros con longitud válida: {len(df_limpio)}")
    print(f"   Registros eliminados: {len(df) - len(df_limpio)}\n")
else:
    df_limpio = df
    print("   ⚠️  No se encontró columna de longitud\n")

# ==================== CALCULAR ESTADÍSTICAS ====================
print("="*60)
print("📊 RESULTADOS DEL ANÁLISIS")
print("="*60)

# 1. PLIEGUES TOTALES
pliegues_totales = len(df_limpio)
print(f"\n🗻 PLIEGUES TOTALES: {pliegues_totales}")

# 2. PLIEGUES LARGOS (> longitud promedio)
pliegues_largos = 0
longitud_promedio = 0

if columna_longitud and not df_limpio[columna_longitud].isna().all():
    longitud_promedio = df_limpio[columna_longitud].mean()
    pliegues_largos = len(df_limpio[df_limpio[columna_longitud] > longitud_promedio])
    print(f"📏 LONGITUD PROMEDIO: {longitud_promedio:.2f} unidades")
    print(f"📐 PLIEGUES LARGOS (> promedio): {pliegues_largos}")
    porcentaje_largos = (pliegues_largos / pliegues_totales * 100) if pliegues_totales > 0 else 0
    print(f"   ({porcentaje_largos:.1f}% del total)")
else:
    print("⚠️  No se pudo calcular pliegues largos (columna de longitud no disponible)")

# 3. DENSIDAD DE PLIEGUES
densidad_pliegues = 0

if columna_area and columna_area in df_limpio.columns:
    # Si hay columna de área, calcular densidad
    df_limpio[columna_area] = pd.to_numeric(df_limpio[columna_area], errors='coerce')
    area_total = df_limpio[columna_area].sum() / 1000000  # Convertir a km²
    if area_total > 0:
        densidad_pliegues = pliegues_totales / area_total
        print(f"\n📍 ÁREA TOTAL: {area_total:.2f} km²")
        print(f"📊 DENSIDAD DE PLIEGUES: {densidad_pliegues:.4f} pliegues/km²")
else:
    # Calcular área aproximada desde coordenadas si están disponibles
    if all(col in df_limpio.columns for col in ['Lat(°)', 'Long(°)']) or \
       all(col in df_limpio.columns for col in ['lat', 'lon']) or \
       all(col in df_limpio.columns for col in ['latitude', 'longitude']):
        
        # Intentar encontrar columnas de coordenadas
        lat_col = next((col for col in df_limpio.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in df_limpio.columns if 'lon' in col.lower()), None)
        
        if lat_col and lon_col:
            df_limpio[lat_col] = pd.to_numeric(df_limpio[lat_col], errors='coerce')
            df_limpio[lon_col] = pd.to_numeric(df_limpio[lon_col], errors='coerce')
            
            # Calcular área aproximada del rectángulo que contiene todos los pliegues
            lat_range = df_limpio[lat_col].max() - df_limpio[lat_col].min()
            lon_range = df_limpio[lon_col].max() - df_limpio[lon_col].min()
            
            # Aproximación: 1 grado ≈ 111 km
            area_aprox = lat_range * lon_range * (111 ** 2)
            
            if area_aprox > 0:
                densidad_pliegues = pliegues_totales / area_aprox
                print(f"\n📍 ÁREA APROXIMADA: {area_aprox:.2f} km²")
                print(f"📊 DENSIDAD DE PLIEGUES: {densidad_pliegues:.4f} pliegues/km²")
    else:
        print("\n⚠️  No se pudo calcular densidad (no hay información de área o coordenadas)")

# 4. TIPO DE PLIEGUE PREDOMINANTE
tipo_pliegue_predominante = "No disponible"

if columna_tipo:
    tipo_counts = df_limpio[columna_tipo].value_counts()
    if len(tipo_counts) > 0:
        tipo_pliegue_predominante = tipo_counts.index[0]
        print(f"\n🏔️  TIPO DE PLIEGUE PREDOMINANTE: {tipo_pliegue_predominante}")
        print(f"   Cantidad: {tipo_counts.iloc[0]} ({tipo_counts.iloc[0]/pliegues_totales*100:.1f}%)")
        
        print(f"\n📊 DISTRIBUCIÓN POR TIPO DE PLIEGUE:")
        for tipo, count in tipo_counts.head(10).items():
            porcentaje = (count / pliegues_totales * 100) if pliegues_totales > 0 else 0
            print(f"   {tipo}: {count} ({porcentaje:.1f}%)")
else:
    print("\n⚠️  No se encontró columna de tipo de pliegue")

# ==================== ESTADÍSTICAS ADICIONALES ====================
if columna_longitud and not df_limpio[columna_longitud].isna().all():
    print(f"\n📊 ESTADÍSTICAS DE LONGITUD:")
    print(f"   Mínima: {df_limpio[columna_longitud].min():.2f} unidades")
    print(f"   Máxima: {df_limpio[columna_longitud].max():.2f} unidades")
    print(f"   Promedio: {df_limpio[columna_longitud].mean():.2f} unidades")
    print(f"   Mediana: {df_limpio[columna_longitud].median():.2f} unidades")
    print(f"   Desviación estándar: {df_limpio[columna_longitud].std():.2f} unidades")

# ==================== RESUMEN FINAL ====================
print("\n" + "="*60)
print("📋 RESUMEN FINAL")
print("="*60)
print(f"✅ Pliegues totales: {pliegues_totales}")
print(f"✅ Pliegues largos (> promedio): {pliegues_largos}")
print(f"✅ Densidad de pliegues: {densidad_pliegues:.4f} pliegues/km²")
print(f"✅ Tipo predominante: {tipo_pliegue_predominante}")
print("="*60)

# ==================== GUARDAR RESULTADOS ====================
print("\n💾 Guardando resultados...")

# Crear un diccionario con los resultados
resultados = {
    'pliegues_totales': pliegues_totales,
    'pliegues_largos': pliegues_largos,
    'densidad_pliegues': densidad_pliegues,
    'tipo_pliegue_predominante': tipo_pliegue_predominante,
    'longitud_promedio': longitud_promedio if columna_longitud else None
}

# Crear DataFrame con los resultados
df_resultados = pd.DataFrame([resultados])

# Guardar como CSV en la carpeta Procesados
archivo_resultados_csv = os.path.join(carpeta_resultados, 'resultados_analisis_pliegues.csv')
df_resultados.to_csv(archivo_resultados_csv, index=False, encoding='utf-8')

# También guardar como archivo de texto en Procesados
archivo_resultados_txt = os.path.join(carpeta_resultados, 'resultados_analisis_pliegues.txt')
with open(archivo_resultados_txt, 'w', encoding='utf-8') as f:
    f.write("ANÁLISIS DE DATOS DE PLIEGUES\n")
    f.write("="*60 + "\n\n")
    f.write(f"Pliegues totales: {pliegues_totales}\n")
    f.write(f"Pliegues largos (> promedio): {pliegues_largos}\n")
    f.write(f"Densidad de pliegues: {densidad_pliegues:.4f} pliegues/km²\n")
    f.write(f"Tipo predominante: {tipo_pliegue_predominante}\n")
    if columna_longitud:
        f.write(f"Longitud promedio: {longitud_promedio:.2f} unidades\n")

print(f"✅ Resultados guardados en la carpeta 'Data/Procesados':")
print(f"   📄 {archivo_resultados_csv}")
print(f"   📄 {archivo_resultados_txt}")
print("\n🎉 ¡Análisis completado con éxito!")