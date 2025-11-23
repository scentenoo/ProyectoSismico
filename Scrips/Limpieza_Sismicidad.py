import pandas as pd
import numpy as np
import re

# ==================== CARGAR ARCHIVO ====================
archivo = 'datos_sismicos.xlsx'  # O 'datos_sismicos.csv' si es CSV
df = pd.read_excel(archivo)  # Si es CSV usa: pd.read_csv(archivo)

print("="*70)
print("🌍 INICIANDO LIMPIEZA DE BASE DE DATOS SÍSMICOS - COLOMBIA")
print("="*70)
print(f"📊 Datos originales: {len(df)} filas, {len(df.columns)} columnas\n")

# ==================== 1. RENOMBRAR COLUMNAS (por si tienen espacios) ====================
print("📝 Paso 1: Estandarizando nombres de columnas...")
df.columns = df.columns.str.strip()
print(f"   Columnas: {list(df.columns)}")
print("   ✅ Completado\n")

# ==================== 2. LIMPIAR ESPACIOS EN TODAS LAS COLUMNAS ====================
print("🧹 Paso 2: Limpiando espacios en blanco...")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()
        # Eliminar espacios múltiples
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
print("   ✅ Completado\n")

# ==================== 3. LIMPIAR Y VALIDAR FECHAS ====================
print("📅 Paso 3: Procesando fechas...")
fecha_col = df.columns[0]  # Primera columna (Fecha-Hora)
try:
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
    filas_antes = len(df)
    df = df.dropna(subset=[fecha_col])
    eliminadas = filas_antes - len(df)
    print(f"   ✅ Fechas procesadas ({eliminadas} fechas inválidas eliminadas)\n")
except Exception as e:
    print(f"   ⚠️  Error procesando fechas: {e}\n")

# ==================== 4. FUNCIÓN PARA LIMPIAR VALORES NUMÉRICOS ====================
def limpiar_numerico(valor):
    """Limpia valores numéricos que pueden tener texto o espacios"""
    if pd.isna(valor):
        return np.nan
    
    # Convertir a string
    valor_str = str(valor)
    
    # Remover espacios
    valor_str = valor_str.strip()
    
    # Extraer solo el primer número (antes de cualquier texto)
    match = re.search(r'^-?\d+\.?\d*', valor_str)
    if match:
        try:
            return float(match.group())
        except:
            return np.nan
    return np.nan

# ==================== 5. LIMPIAR COORDENADAS ====================
print("📍 Paso 4: Limpiando coordenadas...")
columnas_coord = ['Lat(°)', 'Long(°)']
for col in columnas_coord:
    if col in df.columns:
        df[col] = df[col].apply(limpiar_numerico)
        df[col] = df[col].round(4)

# Validar rangos de coordenadas para Colombia
if 'Lat(°)' in df.columns and 'Long(°)' in df.columns:
    filas_antes = len(df)
    df = df[(df['Lat(°)'].between(-5, 14)) & (df['Long(°)'].between(-80, -65))]
    eliminadas = filas_antes - len(df)
    print(f"   ✅ Coordenadas limpiadas ({eliminadas} fuera de rango eliminadas)\n")

# ==================== 6. LIMPIAR TODAS LAS COLUMNAS NUMÉRICAS ====================
print("🔢 Paso 5: Limpiando columnas numéricas...")
columnas_numericas = ['Prof(Km)', 'Mag.', 'Rms(Seg)', "Gap(°)", 
                      'Error Lat(Km)', 'Error Long(Km)', 'Error Prof(Km)']

for col in columnas_numericas:
    if col in df.columns:
        # Aplicar limpieza
        df[col] = df[col].apply(limpiar_numerico)
        
        # Redondear según el tipo de dato
        if col == 'Prof(Km)':
            df[col] = df[col].round(1)
        elif col in ['Mag.', 'Rms(Seg)']:
            df[col] = df[col].round(2)
        elif 'Error' in col:
            df[col] = df[col].round(2)
        elif col == "Gap(°)":
            df[col] = df[col].round(0)

print("   ✅ Completado\n")

# ==================== 7. VALIDAR MAGNITUDES ====================
print("⚡ Paso 6: Validando magnitudes sísmicas...")
if 'Mag.' in df.columns:
    filas_antes = len(df)
    # Mantener solo magnitudes válidas (0-10)
    df = df[(df['Mag.'].isna()) | ((df['Mag.'] >= 0) & (df['Mag.'] <= 10))]
    eliminadas = filas_antes - len(df)
    print(f"   ✅ Magnitudes validadas ({eliminadas} valores inválidos eliminados)\n")

# ==================== 8. VALIDAR PROFUNDIDADES ====================
print("🌊 Paso 7: Validando profundidades...")
if 'Prof(Km)' in df.columns:
    filas_antes = len(df)
    # Mantener solo profundidades positivas y razonables (<700 km)
    df = df[(df['Prof(Km)'].isna()) | ((df['Prof(Km)'] >= 0) & (df['Prof(Km)'] <= 700))]
    eliminadas = filas_antes - len(df)
    print(f"   ✅ Profundidades validadas ({eliminadas} valores inválidos eliminados)\n")

# ==================== 9. ESTANDARIZAR TIPO DE MAGNITUD ====================
print("📊 Paso 8: Estandarizando tipos de magnitud...")
if 'Tipo Mag.' in df.columns:
    df['Tipo Mag.'] = df['Tipo Mag.'].str.replace('MLr_vmm', 'ML', regex=False)
    df['Tipo Mag.'] = df['Tipo Mag.'].str.replace('MLv_vmm', 'ML', regex=False)
    df['Tipo Mag.'] = df['Tipo Mag.'].str.upper()
    df['Tipo Mag.'] = df['Tipo Mag.'].str.strip()
print("   ✅ Completado\n")

# ==================== 10. LIMPIAR REGIONES ====================
print("🗺️  Paso 9: Limpiando nombres de regiones...")
if 'Region' in df.columns:
    # Limpiar y estandarizar
    df['Region'] = df['Region'].str.strip()
    df['Region'] = df['Region'].str.replace(r'\s+', ' ', regex=True)
    # Capitalizar correctamente
    df['Region'] = df['Region'].str.title()
print("   ✅ Completado\n")

# ==================== 11. ELIMINAR DUPLICADOS ====================
print("🔍 Paso 10: Eliminando duplicados...")
filas_antes = len(df)
# Considerar duplicado si fecha, coordenadas y magnitud son iguales
columnas_clave = [fecha_col, 'Lat(°)', 'Long(°)', 'Mag.']
columnas_clave = [col for col in columnas_clave if col in df.columns]
df = df.drop_duplicates(subset=columnas_clave, keep='first')
duplicados = filas_antes - len(df)
print(f"   ✅ {duplicados} eventos duplicados eliminados\n")

# ==================== 12. ELIMINAR FILAS CON DATOS CRÍTICOS VACÍOS ====================
print("🗑️  Paso 11: Eliminando filas con valores críticos vacíos...")
filas_antes = len(df)
columnas_criticas = ['Lat(°)', 'Long(°)']
columnas_criticas = [col for col in columnas_criticas if col in df.columns]
df = df.dropna(subset=columnas_criticas)
vacias = filas_antes - len(df)
print(f"   ✅ {vacias} filas con coordenadas vacías eliminadas\n")

# ==================== 13. ORDENAR POR FECHA ====================
print("📅 Paso 12: Ordenando por fecha (más reciente primero)...")
df = df.sort_values(by=fecha_col, ascending=False)
df = df.reset_index(drop=True)
print("   ✅ Completado\n")

# ==================== RESUMEN FINAL ====================
print("="*70)
print("📈 RESUMEN DE LIMPIEZA")
print("="*70)
print(f"✨ Filas finales: {len(df)}")
print(f"✨ Columnas: {len(df.columns)}")
print(f"✨ Duplicados restantes: {df.duplicated().sum()}")

print(f"\n📊 Valores nulos por columna:")
nulos = df.isnull().sum()
if nulos.sum() > 0:
    print(nulos[nulos > 0])
else:
    print("   ¡No hay valores nulos!")

# Estadísticas de magnitud
if 'Mag.' in df.columns:
    mag_validas = df['Mag.'].dropna()
    if len(mag_validas) > 0:
        print(f"\n⚡ ESTADÍSTICAS DE MAGNITUD:")
        print(f"   Total eventos con magnitud: {len(mag_validas)}")
        print(f"   Rango: {mag_validas.min():.1f} - {mag_validas.max():.1f}")
        print(f"   Promedio: {mag_validas.mean():.2f}")
        print(f"   Mediana: {mag_validas.median():.2f}")

# Estadísticas de profundidad
if 'Prof(Km)' in df.columns:
    prof_validas = df['Prof(Km)'].dropna()
    if len(prof_validas) > 0:
        print(f"\n🌊 ESTADÍSTICAS DE PROFUNDIDAD:")
        print(f"   Rango: {prof_validas.min():.1f} - {prof_validas.max():.1f} km")
        print(f"   Promedio: {prof_validas.mean():.1f} km")
        print(f"   Mediana: {prof_validas.median():.1f} km")

# Distribución por tipo
if 'Tipo Mag.' in df.columns:
    print(f"\n📊 DISTRIBUCIÓN POR TIPO DE MAGNITUD:")
    print(df['Tipo Mag.'].value_counts().head(10))

# Top regiones
if 'Region' in df.columns:
    print(f"\n🗺️  TOP 10 REGIONES CON MÁS EVENTOS:")
    print(df['Region'].value_counts().head(10))

# ==================== GUARDAR ARCHIVO LIMPIO ====================
archivo_limpio = 'datos_sismicos_limpio.xlsx'
df.to_excel(archivo_limpio, index=False)

print("\n" + "="*70)
print(f"💾 ARCHIVO GUARDADO: {archivo_limpio}")
print("="*70)

# ==================== VISTA PREVIA ====================
print("\n📋 VISTA PREVIA (primeras 10 filas):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(df.head(10))

print("\n" + "="*70)
print("✅ ¡LIMPIEZA COMPLETADA CON ÉXITO!")
print("🌍 Base de datos sísmicos lista para análisis")
print("="*70)

