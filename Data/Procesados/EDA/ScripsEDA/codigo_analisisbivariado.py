import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== CONFIGURACIÓN ====================
archivo_sismos = 'Data/Originals/Catálogo Sismicidad TECTO.csv'
archivo_pliegues = 'Data/Procesados/pliegues_limpios.csv'
carpeta_resultados = 'Data/Procesados/EDA'

# Crear carpeta si no existe
os.makedirs(carpeta_resultados, exist_ok=True)

print("="*70)
print("📊 FASE 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("   3.2 ANÁLISIS BIVARIADO")
print("   Correlación: Densidad de Pliegues vs Frecuencia Sísmica")
print("="*70)

# ==================== CARGAR DATOS ====================
print("\n📥 Cargando datos...")

# Cargar sismos
try:
    df_sismos = pd.read_csv(archivo_sismos)
    print(f"✅ Sismos cargados: {len(df_sismos)} registros")
except Exception as e:
    print(f"❌ Error cargando sismos: {e}")
    exit()

# Cargar pliegues
try:
    df_pliegues = pd.read_csv(archivo_pliegues)
    print(f"✅ Pliegues cargados: {len(df_pliegues)} registros")
except Exception as e:
    print(f"❌ Error cargando pliegues: {e}")
    exit()

# ==================== IDENTIFICAR COLUMNAS CLAVE ====================
print("\n🔍 Identificando columnas clave...")

# Columnas de coordenadas para sismos
lat_sismos = None
lon_sismos = None
for col in df_sismos.columns:
    if 'lat' in col.lower() and lat_sismos is None:
        lat_sismos = col
    if 'lon' in col.lower() and lon_sismos is None:
        lon_sismos = col

# Columnas de coordenadas para pliegues
lat_pliegues = None
lon_pliegues = None
for col in df_pliegues.columns:
    if 'lat' in col.lower() and lat_pliegues is None:
        lat_pliegues = col
    if 'lon' in col.lower() and lon_pliegues is None:
        lon_pliegues = col

print(f"   Sismos - Lat: {lat_sismos}, Lon: {lon_sismos}")
print(f"   Pliegues - Lat: {lat_pliegues}, Lon: {lon_pliegues}")

if not all([lat_sismos, lon_sismos, lat_pliegues, lon_pliegues]):
    print("⚠️  No se encontraron todas las columnas de coordenadas necesarias")
    exit()

# ==================== LIMPIAR COORDENADAS ====================
print("\n🧹 Limpiando datos de coordenadas...")

# Limpiar sismos
df_sismos[lat_sismos] = pd.to_numeric(df_sismos[lat_sismos], errors='coerce')
df_sismos[lon_sismos] = pd.to_numeric(df_sismos[lon_sismos], errors='coerce')
df_sismos_limpio = df_sismos.dropna(subset=[lat_sismos, lon_sismos])

# Limpiar pliegues
df_pliegues[lat_pliegues] = pd.to_numeric(df_pliegues[lat_pliegues], errors='coerce')
df_pliegues[lon_pliegues] = pd.to_numeric(df_pliegues[lon_pliegues], errors='coerce')
df_pliegues_limpio = df_pliegues.dropna(subset=[lat_pliegues, lon_pliegues])

print(f"   Sismos válidos: {len(df_sismos_limpio)}")
print(f"   Pliegues válidos: {len(df_pliegues_limpio)}")

# ==================== CREAR GRILLA ESPACIAL ====================
print("\n🗺️  Creando grilla espacial para análisis...")

# Definir límites de la región
lat_min = min(df_sismos_limpio[lat_sismos].min(), df_pliegues_limpio[lat_pliegues].min())
lat_max = max(df_sismos_limpio[lat_sismos].max(), df_pliegues_limpio[lat_pliegues].max())
lon_min = min(df_sismos_limpio[lon_sismos].min(), df_pliegues_limpio[lon_pliegues].min())
lon_max = max(df_sismos_limpio[lon_sismos].max(), df_pliegues_limpio[lon_pliegues].max())

print(f"   Región de estudio:")
print(f"   Latitud: [{lat_min:.2f}, {lat_max:.2f}]")
print(f"   Longitud: [{lon_min:.2f}, {lon_max:.2f}]")

# Crear grilla (celdas de aproximadamente 0.5 grados)
n_celdas = 20
lat_bins = np.linspace(lat_min, lat_max, n_celdas)
lon_bins = np.linspace(lon_min, lon_max, n_celdas)

print(f"   Grilla creada: {n_celdas}x{n_celdas} celdas")

# ==================== CALCULAR DENSIDAD DE PLIEGUES POR CELDA ====================
print("\n📊 Calculando densidad de pliegues por celda...")

# Asignar cada pliegue a una celda
df_pliegues_limpio['celda_lat'] = pd.cut(df_pliegues_limpio[lat_pliegues], bins=lat_bins, labels=False)
df_pliegues_limpio['celda_lon'] = pd.cut(df_pliegues_limpio[lon_pliegues], bins=lon_bins, labels=False)

# Contar pliegues por celda
densidad_pliegues = df_pliegues_limpio.groupby(['celda_lat', 'celda_lon']).size().reset_index(name='num_pliegues')

# Calcular área aproximada de cada celda (km²)
lat_diff = (lat_max - lat_min) / n_celdas
lon_diff = (lon_max - lon_min) / n_celdas
area_celda = (lat_diff * 111) * (lon_diff * 111 * np.cos(np.radians((lat_min + lat_max) / 2)))

# Calcular densidad (pliegues/km²)
densidad_pliegues['densidad_pliegues'] = densidad_pliegues['num_pliegues'] / area_celda

print(f"   Celdas con pliegues: {len(densidad_pliegues)}")
print(f"   Densidad promedio: {densidad_pliegues['densidad_pliegues'].mean():.4f} pliegues/km²")

# ==================== CALCULAR FRECUENCIA SÍSMICA POR CELDA ====================
print("\n📊 Calculando frecuencia sísmica por celda...")

# Asignar cada sismo a una celda
df_sismos_limpio['celda_lat'] = pd.cut(df_sismos_limpio[lat_sismos], bins=lat_bins, labels=False)
df_sismos_limpio['celda_lon'] = pd.cut(df_sismos_limpio[lon_sismos], bins=lon_bins, labels=False)

# Contar sismos por celda
frecuencia_sismos = df_sismos_limpio.groupby(['celda_lat', 'celda_lon']).size().reset_index(name='num_sismos')

# Calcular frecuencia (sismos/km²)
frecuencia_sismos['frecuencia_sismica'] = frecuencia_sismos['num_sismos'] / area_celda

print(f"   Celdas con sismos: {len(frecuencia_sismos)}")
print(f"   Frecuencia promedio: {frecuencia_sismos['frecuencia_sismica'].mean():.4f} sismos/km²")

# ==================== COMBINAR DATOS ====================
print("\n🔗 Combinando datos de pliegues y sismos...")

# Merge de densidad de pliegues y frecuencia sísmica
df_combinado = pd.merge(densidad_pliegues, frecuencia_sismos, 
                        on=['celda_lat', 'celda_lon'], how='outer')

# Rellenar NaN con 0 (celdas sin datos)
df_combinado['densidad_pliegues'] = df_combinado['densidad_pliegues'].fillna(0)
df_combinado['frecuencia_sismica'] = df_combinado['frecuencia_sismica'].fillna(0)
df_combinado['num_pliegues'] = df_combinado['num_pliegues'].fillna(0)
df_combinado['num_sismos'] = df_combinado['num_sismos'].fillna(0)

print(f"   Total de celdas analizadas: {len(df_combinado)}")
print(f"   Celdas con ambos (pliegues y sismos): {len(df_combinado[(df_combinado['num_pliegues'] > 0) & (df_combinado['num_sismos'] > 0)])}")

# ==================== ANÁLISIS DE CORRELACIÓN ====================
print("\n" + "="*70)
print("📈 ANÁLISIS DE CORRELACIÓN")
print("="*70)

# Filtrar solo celdas con datos (al menos un pliegue o un sismo)
df_analisis = df_combinado[(df_combinado['num_pliegues'] > 0) | (df_combinado['num_sismos'] > 0)]

# Calcular correlación de Pearson
correlacion_pearson, p_valor_pearson = stats.pearsonr(df_analisis['densidad_pliegues'], 
                                                       df_analisis['frecuencia_sismica'])

# Calcular correlación de Spearman (no paramétrica)
correlacion_spearman, p_valor_spearman = stats.spearmanr(df_analisis['densidad_pliegues'], 
                                                          df_analisis['frecuencia_sismica'])

print(f"\n📊 Resultados de Correlación:")
print(f"   Correlación de Pearson: {correlacion_pearson:.4f} (p-valor: {p_valor_pearson:.4f})")
print(f"   Correlación de Spearman: {correlacion_spearman:.4f} (p-valor: {p_valor_spearman:.4f})")

# Interpretación
if abs(correlacion_pearson) < 0.3:
    interpretacion = "débil o inexistente"
elif abs(correlacion_pearson) < 0.7:
    interpretacion = "moderada"
else:
    interpretacion = "fuerte"

direccion = "positiva" if correlacion_pearson > 0 else "negativa"

print(f"\n💡 Interpretación:")
print(f"   La correlación es {interpretacion} y {direccion}.")

if p_valor_pearson < 0.05:
    print(f"   ✅ La correlación es estadísticamente significativa (p < 0.05)")
else:
    print(f"   ⚠️  La correlación NO es estadísticamente significativa (p >= 0.05)")

# ==================== VISUALIZACIONES ====================
print("\n📊 Generando visualizaciones...")

# Crear figura con múltiples gráficos
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Análisis Bivariado: Densidad de Pliegues vs Frecuencia Sísmica', 
             fontsize=18, fontweight='bold')

# 1. Gráfico de dispersión principal
ax1 = fig.add_subplot(gs[0:2, 0:2])
scatter = ax1.scatter(df_analisis['densidad_pliegues'], 
                     df_analisis['frecuencia_sismica'],
                     c=df_analisis['num_sismos'], 
                     s=100, 
                     alpha=0.6, 
                     cmap='YlOrRd',
                     edgecolors='black',
                     linewidth=0.5)

# Línea de tendencia
z = np.polyfit(df_analisis['densidad_pliegues'], df_analisis['frecuencia_sismica'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_analisis['densidad_pliegues'].min(), 
                     df_analisis['densidad_pliegues'].max(), 100)
ax1.plot(x_line, p(x_line), "r--", linewidth=2, label='Línea de tendencia')

ax1.set_xlabel('Densidad de Pliegues (pliegues/km²)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frecuencia Sísmica (sismos/km²)', fontsize=12, fontweight='bold')
ax1.set_title(f'Correlación de Pearson: {correlacion_pearson:.4f} (p={p_valor_pearson:.4f})', 
              fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Colorbar
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Número de Sismos', rotation=270, labelpad=20)

# 2. Histograma de densidad de pliegues
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df_analisis['densidad_pliegues'], bins=30, color='steelblue', 
         edgecolor='black', alpha=0.7)
ax2.set_xlabel('Densidad de Pliegues', fontsize=10)
ax2.set_ylabel('Frecuencia', fontsize=10)
ax2.set_title('Distribución de\nDensidad de Pliegues', fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. Histograma de frecuencia sísmica
ax3 = fig.add_subplot(gs[1, 2])
ax3.hist(df_analisis['frecuencia_sismica'], bins=30, color='coral', 
         edgecolor='black', alpha=0.7)
ax3.set_xlabel('Frecuencia Sísmica', fontsize=10)
ax3.set_ylabel('Frecuencia', fontsize=10)
ax3.set_title('Distribución de\nFrecuencia Sísmica', fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Mapa de calor - Densidad de pliegues
ax4 = fig.add_subplot(gs[2, 0])
matriz_pliegues = np.zeros((n_celdas-1, n_celdas-1))
for _, row in df_combinado.iterrows():
    if not pd.isna(row['celda_lat']) and not pd.isna(row['celda_lon']):
        matriz_pliegues[int(row['celda_lat']), int(row['celda_lon'])] = row['densidad_pliegues']

im1 = ax4.imshow(matriz_pliegues, cmap='Blues', origin='lower', aspect='auto')
ax4.set_xlabel('Longitud (celda)', fontsize=10)
ax4.set_ylabel('Latitud (celda)', fontsize=10)
ax4.set_title('Mapa de Calor:\nDensidad de Pliegues', fontsize=11)
plt.colorbar(im1, ax=ax4, label='Densidad')

# 5. Mapa de calor - Frecuencia sísmica
ax5 = fig.add_subplot(gs[2, 1])
matriz_sismos = np.zeros((n_celdas-1, n_celdas-1))
for _, row in df_combinado.iterrows():
    if not pd.isna(row['celda_lat']) and not pd.isna(row['celda_lon']):
        matriz_sismos[int(row['celda_lat']), int(row['celda_lon'])] = row['frecuencia_sismica']

im2 = ax5.imshow(matriz_sismos, cmap='Reds', origin='lower', aspect='auto')
ax5.set_xlabel('Longitud (celda)', fontsize=10)
ax5.set_ylabel('Latitud (celda)', fontsize=10)
ax5.set_title('Mapa de Calor:\nFrecuencia Sísmica', fontsize=11)
plt.colorbar(im2, ax=ax5, label='Frecuencia')

# 6. Tabla de estadísticas
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

estadisticas = [
    ['Métrica', 'Valor'],
    ['', ''],
    ['Correlación Pearson', f'{correlacion_pearson:.4f}'],
    ['P-valor Pearson', f'{p_valor_pearson:.4f}'],
    ['Correlación Spearman', f'{correlacion_spearman:.4f}'],
    ['P-valor Spearman', f'{p_valor_spearman:.4f}'],
    ['', ''],
    ['Celdas analizadas', f'{len(df_analisis)}'],
    ['Pliegues totales', f'{int(df_combinado["num_pliegues"].sum())}'],
    ['Sismos totales', f'{int(df_combinado["num_sismos"].sum())}'],
]

tabla = ax6.table(cellText=estadisticas, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 2)
ax6.set_title('Estadísticas del Análisis', fontsize=11, fontweight='bold', pad=20)

# Guardar figura
archivo_bivariado = os.path.join(carpeta_resultados, '4_analisis_bivariado_correlacion.png')
plt.savefig(archivo_bivariado, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {archivo_bivariado}")
plt.close()

# ==================== GUARDAR RESULTADOS ====================
print("\n💾 Guardando resultados en CSV...")

# Guardar datos combinados
archivo_datos = os.path.join(carpeta_resultados, 'datos_bivariado_pliegues_sismos.csv')
df_combinado.to_csv(archivo_datos, index=False, encoding='utf-8')
print(f"✅ Datos guardados: {archivo_datos}")

# Guardar resumen de correlación
resumen = {
    'correlacion_pearson': [correlacion_pearson],
    'p_valor_pearson': [p_valor_pearson],
    'correlacion_spearman': [correlacion_spearman],
    'p_valor_spearman': [p_valor_spearman],
    'interpretacion': [f'{interpretacion} {direccion}'],
    'significancia': ['Sí' if p_valor_pearson < 0.05 else 'No'],
    'celdas_analizadas': [len(df_analisis)],
    'area_celda_km2': [area_celda]
}

df_resumen = pd.DataFrame(resumen)
archivo_resumen = os.path.join(carpeta_resultados, 'resumen_correlacion.csv')
df_resumen.to_csv(archivo_resumen, index=False, encoding='utf-8')
print(f"✅ Resumen guardado: {archivo_resumen}")

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print("✅ ANÁLISIS BIVARIADO COMPLETADO")
print("="*70)
print(f"\n📁 Archivos generados en: {carpeta_resultados}")
print("   📊 4_analisis_bivariado_correlacion.png")
print("   📄 datos_bivariado_pliegues_sismos.csv")
print("   📄 resumen_correlacion.csv")
print("\n🎉 ¡Análisis completado con éxito!")