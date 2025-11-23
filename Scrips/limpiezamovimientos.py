import pandas as pd
import numpy as np

# ==================== CARGAR ARCHIVO ====================
archivo = 'Inventario_de_movimientos_e_0.xlsx'
df = pd.read_excel(archivo)

print("="*60)
print("🚀 INICIANDO LIMPIEZA DE BASE DE DATOS")
print("="*60)
print(f"📊 Datos originales: {len(df)} filas, {len(df.columns)} columnas\n")

# ==================== 1. LIMPIAR ESPACIOS ====================
print("🧹 Paso 1: Limpiando espacios en blanco...")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()
print("   ✅ Completado\n")

# ==================== 2. CORREGIR OBJECTID ====================
print("🔧 Paso 2: Corrigiendo formato de OBJECTID...")
if 'OBJECTID' in df.columns:
    # Convertir notación científica a números enteros
    df['OBJECTID'] = pd.to_numeric(df['OBJECTID'], errors='coerce')
    df['OBJECTID'] = df['OBJECTID'].astype('Int64')
print("   ✅ Completado\n")

# ==================== 3. ESTANDARIZAR NOMBRES ====================
print("📝 Paso 3: Estandarizando nombres de movimientos...")
# Correcciones en Tipo_Movimiento
correcciones_tipo = {
    'Reptació': 'Reptación',
    'Volcamie': 'Volcamiento',
    'Reptació Ir': 'Reptación Irregular',
    'Volcamie vr': 'Volcamiento'
}

# Correcciones en Etiqueta (abreviaturas)
correcciones_etiqueta = {
    'Deslizam dt': 'Deslizamiento dt',
    'Deslizam dtp': 'Deslizamiento dtp',
    'Deslizam dr': 'Deslizamiento dr',
    'Reptació Ir': 'Reptación Irregular',
    'Volcamie vr': 'Volcamiento vr',
    'Avalanch ar': 'Avalancha ar'
}

# Aplicar correcciones
for col in ['Tipo_Movimiento', 'Subtipo_Movimiento', 'Subtipo_nombre', 'Etiqueta']:
    if col in df.columns:
        df[col] = df[col].replace(correcciones_tipo)
        df[col] = df[col].replace(correcciones_etiqueta)
print("   ✅ Completado\n")

# ==================== 4. LIMPIAR COORDENADAS ====================
print("📍 Paso 4: Optimizando coordenadas...")
if 'x' in df.columns and 'y' in df.columns:
    # Redondear a 6 decimales (suficiente precisión para coordenadas)
    df['x'] = pd.to_numeric(df['x'], errors='coerce').round(6)
    df['y'] = pd.to_numeric(df['y'], errors='coerce').round(6)
    
    # Eliminar filas con coordenadas inválidas
    filas_antes = len(df)
    df = df.dropna(subset=['x', 'y'])
    eliminadas = filas_antes - len(df)
    print(f"   ✅ Coordenadas optimizadas ({eliminadas} filas con coordenadas inválidas eliminadas)\n")

# ==================== 5. ELIMINAR DUPLICADOS ====================
print("🔍 Paso 5: Eliminando duplicados...")
filas_antes = len(df)
df = df.drop_duplicates()
duplicados = filas_antes - len(df)
print(f"   ✅ {duplicados} filas duplicadas eliminadas\n")

# ==================== 6. ELIMINAR FILAS VACÍAS ====================
print("🗑️  Paso 6: Eliminando filas con valores vacíos...")
filas_antes = len(df)
df = df.dropna()
vacias = filas_antes - len(df)
print(f"   ✅ {vacias} filas con valores vacíos eliminadas\n")

# ==================== 7. NORMALIZAR COLUMNAS NUMÉRICAS ====================
print("🔢 Paso 7: Normalizando columnas numéricas...")
columnas_numericas = ['FID', 'ID', 'Inentario', 'F35DOV_', 'Representación_ESRI', 'OID']
for col in columnas_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
print("   ✅ Completado\n")

# ==================== 8. RESETEAR ÍNDICES ====================
df = df.reset_index(drop=True)

# ==================== RESUMEN FINAL ====================
print("="*60)
print("📈 RESUMEN DE LIMPIEZA")
print("="*60)
print(f"✨ Total de filas finales: {len(df)}")
print(f"✨ Total de columnas: {len(df.columns)}")
print(f"✨ Duplicados restantes: {df.duplicated().sum()}")
print(f"✨ Valores nulos totales: {df.isnull().sum().sum()}")

print("\n📊 Distribución de tipos de movimiento:")
if 'Tipo_Movimiento' in df.columns:
    print(df['Tipo_Movimiento'].value_counts().to_string())

print("\n📊 Rango de coordenadas:")
if 'x' in df.columns and 'y' in df.columns:
    print(f"   X: [{df['x'].min()}, {df['x'].max()}]")
    print(f"   Y: [{df['y'].min()}, {df['y'].max()}]")

# ==================== GUARDAR ARCHIVO LIMPIO ====================
archivo_limpio = 'Inventario_limpio.xlsx'
df.to_excel(archivo_limpio, index=False)

print("\n" + "="*60)
print(f"💾 ARCHIVO GUARDADO: {archivo_limpio}")
print("="*60)

# ==================== VISTA PREVIA ====================
print("\n📋 Vista previa (primeras 10 filas):")
print(df.head(10).to_string(max_colwidth=30))

print("\n✅ ¡LIMPIEZA COMPLETADA CON ÉXITO!")