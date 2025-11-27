import pandas as pd
import os

# ==================== CONFIGURACIÓN ====================
# Nombre de tu archivo CSV
archivo = 'Data/Originals/Catálogo Sismicidad TECTO.csv'

# Carpeta donde se guardarán los resultados
carpeta_resultados = 'Data/Procesados'

# Crear la carpeta si no existe
os.makedirs(carpeta_resultados, exist_ok=True)

print("="*60)
print("🌍 ANÁLISIS DE DATOS SÍSMICOS")
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

# ==================== IDENTIFICAR COLUMNA DE MAGNITUD ====================
# Buscar la columna de magnitud (puede tener diferentes nombres)
columna_magnitud = None
posibles_nombres = ['Mag.', 'Mag', 'Magnitud', 'magnitude', 'MAG']

for nombre in posibles_nombres:
    if nombre in df.columns:
        columna_magnitud = nombre
        break

if columna_magnitud is None:
    print("⚠️  No se encontró la columna de magnitud.")
    print("Por favor, indica el nombre exacto de la columna de magnitud.")
    exit()

print(f"✅ Columna de magnitud identificada: '{columna_magnitud}'\n")

# ==================== LIMPIAR DATOS ====================
print("🧹 Limpiando datos de magnitud...")

# Convertir a numérico y eliminar valores nulos
df[columna_magnitud] = pd.to_numeric(df[columna_magnitud], errors='coerce')
df_limpio = df.dropna(subset=[columna_magnitud])

print(f"   Registros con magnitud válida: {len(df_limpio)}")
print(f"   Registros eliminados (sin magnitud): {len(df) - len(df_limpio)}\n")

# ==================== CALCULAR ESTADÍSTICAS ====================
print("="*60)
print("📊 RESULTADOS DEL ANÁLISIS")
print("="*60)

# 1. SISMOS TOTALES
sismos_totales = len(df_limpio)
print(f"\n🌍 SISMOS TOTALES: {sismos_totales}")

# 2. SISMOS CON MAGNITUD ALTA (> 3.0)
sismos_magnitud_alta = len(df_limpio[df_limpio[columna_magnitud] > 3.0])
print(f"⚡ SISMOS CON MAGNITUD > 3.0: {sismos_magnitud_alta}")
porcentaje_alta = (sismos_magnitud_alta / sismos_totales * 100) if sismos_totales > 0 else 0
print(f"   ({porcentaje_alta:.1f}% del total)")

# 3. MAGNITUD MÁXIMA
magnitud_maxima = df_limpio[columna_magnitud].max()
print(f"📈 MAGNITUD MÁXIMA HISTÓRICA: {magnitud_maxima:.2f}")

# Información adicional sobre el evento máximo
evento_maximo = df_limpio[df_limpio[columna_magnitud] == magnitud_maxima].iloc[0]
print(f"\n📍 Detalles del sismo máximo:")
if 'Fecha-Hora (UTC)' in df.columns or 'FechaHora (UTC)' in df.columns:
    fecha_col = 'Fecha-Hora (UTC)' if 'Fecha-Hora (UTC)' in df.columns else 'FechaHora (UTC)'
    print(f"   Fecha: {evento_maximo[fecha_col]}")
if 'Region' in df.columns:
    print(f"   Región: {evento_maximo['Region']}")
if 'Lat(°)' in df.columns and 'Long(°)' in df.columns:
    print(f"   Coordenadas: ({evento_maximo['Lat(°)']}, {evento_maximo['Long(°)']})")
if 'Prof(Km)' in df.columns:
    print(f"   Profundidad: {evento_maximo['Prof(Km)']} km")

# ==================== ESTADÍSTICAS ADICIONALES ====================
print(f"\n📊 ESTADÍSTICAS GENERALES DE MAGNITUD:")
print(f"   Promedio: {df_limpio[columna_magnitud].mean():.2f}")
print(f"   Mediana: {df_limpio[columna_magnitud].median():.2f}")
print(f"   Mínima: {df_limpio[columna_magnitud].min():.2f}")
print(f"   Desviación estándar: {df_limpio[columna_magnitud].std():.2f}")

# ==================== DISTRIBUCIÓN POR RANGOS ====================
print(f"\n📊 DISTRIBUCIÓN POR RANGOS DE MAGNITUD:")
rangos = [
    (0, 2.0, "Menor (< 2.0)"),
    (2.0, 3.0, "Baja (2.0 - 3.0)"),
    (3.0, 4.0, "Moderada (3.0 - 4.0)"),
    (4.0, 5.0, "Significativa (4.0 - 5.0)"),
    (5.0, 100, "Alta (≥ 5.0)")
]

for min_mag, max_mag, etiqueta in rangos:
    count = len(df_limpio[(df_limpio[columna_magnitud] >= min_mag) & (df_limpio[columna_magnitud] < max_mag)])
    porcentaje = (count / sismos_totales * 100) if sismos_totales > 0 else 0
    print(f"   {etiqueta}: {count} ({porcentaje:.1f}%)")

# ==================== RESUMEN FINAL ====================
print("\n" + "="*60)
print("📋 RESUMEN FINAL")
print("="*60)
print(f"✅ Sismos totales: {sismos_totales}")
print(f"✅ Sismos con magnitud > 3.0: {sismos_magnitud_alta}")
print(f"✅ Magnitud máxima histórica: {magnitud_maxima:.2f}")
print("="*60)

# ==================== GUARDAR RESULTADOS ====================
print("\n💾 Guardando resultados...")

# Crear un diccionario con los resultados
resultados = {
    'sismos_totales': sismos_totales,
    'sismos_magnitud_alta': sismos_magnitud_alta,
    'magnitud_maxima': magnitud_maxima,
    'magnitud_promedio': df_limpio[columna_magnitud].mean(),
    'magnitud_mediana': df_limpio[columna_magnitud].median()
}

# Crear DataFrame con los resultados
df_resultados = pd.DataFrame([resultados])

# Guardar como CSV en la carpeta Procesados
archivo_resultados_csv = os.path.join(carpeta_resultados, 'resultados_analisis_sismos.csv')
df_resultados.to_csv(archivo_resultados_csv, index=False, encoding='utf-8')

# También guardar como archivo de texto en Procesados
archivo_resultados_txt = os.path.join(carpeta_resultados, 'resultados_analisis_sismos.txt')
with open(archivo_resultados_txt, 'w', encoding='utf-8') as f:
    f.write("ANÁLISIS DE DATOS SÍSMICOS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Sismos totales: {sismos_totales}\n")
    f.write(f"Sismos con magnitud > 3.0: {sismos_magnitud_alta}\n")
    f.write(f"Magnitud máxima histórica: {magnitud_maxima:.2f}\n")
    f.write(f"Magnitud promedio: {resultados['magnitud_promedio']:.2f}\n")
    f.write(f"Magnitud mediana: {resultados['magnitud_mediana']:.2f}\n")

print(f"✅ Resultados guardados en la carpeta 'Data/Procesados':")
print(f"   📄 {archivo_resultados_csv}")
print(f"   📄 {archivo_resultados_txt}")
print("\n🎉 ¡Análisis completado con éxito!")