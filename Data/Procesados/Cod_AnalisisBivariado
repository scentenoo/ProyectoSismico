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
carpeta_resultados = 'Data/Procesados/EDA'

# Crear carpeta si no existe
os.makedirs(carpeta_resultados, exist_ok=True)

print("="*70)
print("📊 FASE 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("   3.2 ANÁLISIS BIVARIADO - VARIABLES SÍSMICAS")
print("="*70)

# ==================== CARGAR DATOS ====================
print("\n📥 Cargando datos de sismos...")

try:
    df = pd.read_csv(archivo_sismos)
    print(f"✅ Sismos cargados: {len(df)} registros")
    print(f"📋 Columnas: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error cargando sismos: {e}")
    exit()

# ==================== IDENTIFICAR Y LIMPIAR COLUMNAS ====================
print("\n🔍 Identificando columnas clave...")

# Buscar columnas
columna_mag = None
columna_prof = None
columna_fecha = None
columna_region = None
columna_lat = None
columna_lon = None

for col in df.columns:
    if 'mag' in col.lower() and columna_mag is None:
        columna_mag = col
    if 'prof' in col.lower() and columna_prof is None:
        columna_prof = col
    if 'fecha' in col.lower() or 'date' in col.lower():
        columna_fecha = col
    if 'region' in col.lower():
        columna_region = col
    if 'lat' in col.lower() and columna_lat is None:
        columna_lat = col
    if 'lon' in col.lower() and columna_lon is None:
        columna_lon = col

print(f"   Magnitud: {columna_mag}")
print(f"   Profundidad: {columna_prof}")
print(f"   Fecha: {columna_fecha}")
print(f"   Región: {columna_region}")
print(f"   Coordenadas: {columna_lat}, {columna_lon}")

# ==================== LIMPIAR DATOS ====================
print("\n🧹 Limpiando datos...")

if columna_mag:
    df[columna_mag] = pd.to_numeric(df[columna_mag], errors='coerce')
if columna_prof:
    df[columna_prof] = pd.to_numeric(df[columna_prof], errors='coerce')
if columna_lat:
    df[columna_lat] = pd.to_numeric(df[columna_lat], errors='coerce')
if columna_lon:
    df[columna_lon] = pd.to_numeric(df[columna_lon], errors='coerce')
if columna_fecha:
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')

# Filtrar datos válidos
df_limpio = df.dropna(subset=[col for col in [columna_mag, columna_prof] if col is not None])
print(f"✅ Datos limpios: {len(df_limpio)} registros")

# ==================== ANÁLISIS DE CORRELACIÓN ====================
print("\n" + "="*70)
print("📈 ANÁLISIS DE CORRELACIONES")
print("="*70)

# Preparar matriz de correlación
columnas_numericas = [columna_mag, columna_prof]
if columna_lat:
    columnas_numericas.append(columna_lat)
if columna_lon:
    columnas_numericas.append(columna_lon)

columnas_numericas = [col for col in columnas_numericas if col is not None]

# Calcular matriz de correlación
df_corr = df_limpio[columnas_numericas].corr()

print("\n📊 Matriz de Correlación:")
print(df_corr.round(3))

# ==================== VISUALIZACIONES ====================
print("\n📊 Generando visualizaciones...")

# Crear figura grande con múltiples subplots
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

fig.suptitle('Análisis Bivariado - Variables Sísmicas', fontsize=20, fontweight='bold', y=0.995)

# ==================== 1. MAGNITUD VS PROFUNDIDAD ====================
ax1 = fig.add_subplot(gs[0, 0:2])

if columna_mag and columna_prof:
    # Gráfico de dispersión
    scatter = ax1.scatter(df_limpio[columna_prof], df_limpio[columna_mag],
                         c=df_limpio[columna_mag], cmap='viridis',
                         s=50, alpha=0.5, edgecolors='black', linewidth=0.3)
    
    # Línea de tendencia
    mask = ~(df_limpio[columna_prof].isna() | df_limpio[columna_mag].isna())
    if mask.sum() > 1:
        z = np.polyfit(df_limpio[columna_prof][mask], df_limpio[columna_mag][mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_limpio[columna_prof].min(), df_limpio[columna_prof].max(), 100)
        ax1.plot(x_line, p(x_line), "r--", linewidth=2, label='Tendencia')
    
    # Calcular correlación
    corr, pval = stats.pearsonr(df_limpio[columna_prof][mask], df_limpio[columna_mag][mask])
    
    ax1.set_xlabel('Profundidad (km)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Magnitud', fontsize=12, fontweight='bold')
    ax1.set_title(f'Magnitud vs Profundidad (r={corr:.3f}, p={pval:.4f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Magnitud', rotation=270, labelpad=15)

# ==================== 2. MATRIZ DE CORRELACIÓN ====================
ax2 = fig.add_subplot(gs[0, 2])

mask_corr = np.triu(np.ones_like(df_corr, dtype=bool))
sns.heatmap(df_corr, mask=mask_corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax2, vmin=-1, vmax=1)
ax2.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')

# ==================== 3. DISTRIBUCIÓN CONJUNTA MAGNITUD-PROFUNDIDAD ====================
ax3 = fig.add_subplot(gs[1, 0])

if columna_mag and columna_prof:
    ax3.hexbin(df_limpio[columna_prof], df_limpio[columna_mag],
              gridsize=30, cmap='YlOrRd', mincnt=1)
    ax3.set_xlabel('Profundidad (km)', fontsize=11)
    ax3.set_ylabel('Magnitud', fontsize=11)
    ax3.set_title('Densidad: Magnitud vs Profundidad', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

# ==================== 4. BOXPLOT MAGNITUD POR RANGOS DE PROFUNDIDAD ====================
ax4 = fig.add_subplot(gs[1, 1])

if columna_mag and columna_prof:
    # Crear rangos de profundidad
    df_limpio['rango_prof'] = pd.cut(df_limpio[columna_prof], 
                                     bins=[0, 30, 70, 150, 1000],
                                     labels=['0-30km', '30-70km', '70-150km', '>150km'])
    
    df_limpio.boxplot(column=columna_mag, by='rango_prof', ax=ax4)
    ax4.set_xlabel('Rango de Profundidad', fontsize=11)
    ax4.set_ylabel('Magnitud', fontsize=11)
    ax4.set_title('Distribución de Magnitud por Profundidad', fontsize=12, fontweight='bold')
    plt.sca(ax4)
    plt.xticks(rotation=45)
    ax4.get_figure().suptitle('')  # Remover título automático

# ==================== 5. DISTRIBUCIÓN ESPACIAL (LAT vs LON) ====================
ax5 = fig.add_subplot(gs[1, 2])

if columna_lat and columna_lon and columna_mag:
    scatter2 = ax5.scatter(df_limpio[columna_lon], df_limpio[columna_lat],
                          c=df_limpio[columna_mag], cmap='jet',
                          s=30, alpha=0.6, edgecolors='black', linewidth=0.2)
    ax5.set_xlabel('Longitud', fontsize=11)
    ax5.set_ylabel('Latitud', fontsize=11)
    ax5.set_title('Distribución Espacial de Sismos', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax5)
    cbar2.set_label('Magnitud', rotation=270, labelpad=15)

# ==================== 6. SERIE TEMPORAL DE MAGNITUDES ====================
ax6 = fig.add_subplot(gs[2, 0:2])

if columna_fecha and columna_mag:
    df_temporal = df_limpio.dropna(subset=[columna_fecha, columna_mag])
    df_temporal = df_temporal.sort_values(columna_fecha)
    
    ax6.scatter(df_temporal[columna_fecha], df_temporal[columna_mag],
               c=df_temporal[columna_mag], cmap='plasma',
               s=20, alpha=0.5)
    
    # Media móvil
    df_temporal['mag_ma'] = df_temporal[columna_mag].rolling(window=50, center=True).mean()
    ax6.plot(df_temporal[columna_fecha], df_temporal['mag_ma'],
            color='red', linewidth=2, label='Media móvil (50 eventos)')
    
    ax6.set_xlabel('Fecha', fontsize=12)
    ax6.set_ylabel('Magnitud', fontsize=12)
    ax6.set_title('Evolución Temporal de Magnitudes', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

# ==================== 7. HISTOGRAMA 2D MAGNITUD-PROFUNDIDAD ====================
ax7 = fig.add_subplot(gs[2, 2])

if columna_mag and columna_prof:
    hist = ax7.hist2d(df_limpio[columna_prof], df_limpio[columna_mag],
                     bins=30, cmap='hot', cmin=1)
    ax7.set_xlabel('Profundidad (km)', fontsize=11)
    ax7.set_ylabel('Magnitud', fontsize=11)
    ax7.set_title('Histograma 2D', fontsize=12, fontweight='bold')
    plt.colorbar(hist[3], ax=ax7, label='Frecuencia')

# ==================== 8. ANÁLISIS POR REGIÓN (TOP 10) ====================
ax8 = fig.add_subplot(gs[3, 0:2])

if columna_region and columna_mag:
    # Top 10 regiones con más sismos
    top_regiones = df_limpio[columna_region].value_counts().head(10)
    
    # Preparar datos para boxplot
    datos_regiones = []
    labels_regiones = []
    for region in top_regiones.index:
        mags = df_limpio[df_limpio[columna_region] == region][columna_mag].dropna()
        if len(mags) > 0:
            datos_regiones.append(mags)
            # Truncar nombres largos
            label = region[:30] + '...' if len(region) > 30 else region
            labels_regiones.append(label)
    
    bp = ax8.boxplot(datos_regiones, labels=labels_regiones, patch_artist=True)
    
    # Colorear cajas
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax8.set_xlabel('Región', fontsize=12)
    ax8.set_ylabel('Magnitud', fontsize=12)
    ax8.set_title('Distribución de Magnitudes por Región (Top 10)', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ==================== 9. TABLA DE ESTADÍSTICAS ====================
ax9 = fig.add_subplot(gs[3, 2])
ax9.axis('off')

estadisticas = [
    ['Estadística', 'Valor'],
    ['', ''],
    ['Total sismos', f'{len(df_limpio)}'],
    ['', ''],
]

if columna_mag and columna_prof:
    corr_mp, pval_mp = stats.pearsonr(df_limpio[columna_prof].dropna(), 
                                       df_limpio[columna_mag].dropna())
    estadisticas.extend([
        ['Corr. Mag-Prof', f'{corr_mp:.3f}'],
        ['P-valor', f'{pval_mp:.4f}'],
        ['', ''],
    ])

if columna_mag:
    estadisticas.extend([
        ['Mag. promedio', f'{df_limpio[columna_mag].mean():.2f}'],
        ['Mag. mediana', f'{df_limpio[columna_mag].median():.2f}'],
        ['Mag. máxima', f'{df_limpio[columna_mag].max():.2f}'],
        ['', ''],
    ])

if columna_prof:
    estadisticas.extend([
        ['Prof. promedio', f'{df_limpio[columna_prof].mean():.1f} km'],
        ['Prof. mediana', f'{df_limpio[columna_prof].median():.1f} km'],
        ['Prof. máxima', f'{df_limpio[columna_prof].max():.1f} km'],
    ])

tabla = ax9.table(cellText=estadisticas, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 2)
ax9.set_title('Estadísticas Resumen', fontsize=12, fontweight='bold', pad=20)

# Guardar figura
archivo_bivariado = os.path.join(carpeta_resultados, '4_analisis_bivariado_sismos.png')
plt.savefig(archivo_bivariado, dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: {archivo_bivariado}")
plt.close()

# ==================== GUARDAR ANÁLISIS DE CORRELACIÓN ====================
print("\n💾 Guardando resultados de correlación...")

# Guardar matriz de correlación
archivo_corr = os.path.join(carpeta_resultados, 'matriz_correlacion_sismos.csv')
df_corr.to_csv(archivo_corr, encoding='utf-8')
print(f"✅ Matriz guardada: {archivo_corr}")

# Crear resumen de análisis
resumen = []
if columna_mag and columna_prof:
    corr_mp, pval_mp = stats.pearsonr(df_limpio[columna_prof].dropna(), 
                                       df_limpio[columna_mag].dropna())
    resumen.append({
        'Variables': 'Magnitud vs Profundidad',
        'Correlación': corr_mp,
        'P-valor': pval_mp,
        'Significativo': 'Sí' if pval_mp < 0.05 else 'No'
    })

if len(resumen) > 0:
    df_resumen = pd.DataFrame(resumen)
    archivo_resumen = os.path.join(carpeta_resultados, 'resumen_correlaciones_bivariado.csv')
    df_resumen.to_csv(archivo_resumen, index=False, encoding='utf-8')
    print(f"✅ Resumen guardado: {archivo_resumen}")

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print("✅ ANÁLISIS BIVARIADO COMPLETADO")
print("="*70)
print(f"\n📁 Archivos generados en: {carpeta_resultados}")
print("   📊 4_analisis_bivariado_sismos.png")
print("   📄 matriz_correlacion_sismos.csv")
print("   📄 resumen_correlaciones_bivariado.csv")
print("\n🎉 ¡Análisis bivariado completado con éxito!")
