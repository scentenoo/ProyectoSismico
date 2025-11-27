import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np
from shapely.geometry import Point, shape, LineString

print("🗻 ANÁLISIS DE PLIEGUES GEOLÓGICOS - COLOMBIA")
print("=" * 60)

# 1. CARGAR DATOS
print("\n📂 Cargando datos...")

# Sismos
sismos = pd.read_excel('Data/Procesados/LLCatálogo Sismicidad TECTO_limpio.xlsx')
sismos['lat'] = sismos['Lat(°)'] / 1000
sismos['lon'] = sismos['Long(°)'] / 1000
print(f"   ✅ Sismos: {len(sismos)}")

# Pliegues
pliegues = pd.read_csv('Data/Procesados/pliegues_limpios.csv')
print(f"   ✅ Pliegues: {len(pliegues)}")
print(f"   📋 Columnas en pliegues: {list(pliegues.columns)}")

# Municipios
with open('Data/Procesados/MunicipiosColombia.json', 'r', encoding='utf-8') as f:
    municipios = json.load(f)
print(f"   ✅ Municipios: {len(municipios['features'])}")

# 2. ANALIZAR PLIEGUES
print("\n📊 Análisis de pliegues:")
print("=" * 60)

if 'Tipo' in pliegues.columns:
    tipos = pliegues['Tipo'].value_counts()
    print("\n   Tipos de pliegues encontrados:")
    for tipo, cantidad in tipos.items():
        print(f"      {tipo}: {cantidad}")

if 'Shape__Length' in pliegues.columns:
    longitud_total = pliegues['Shape__Length'].sum()
    longitud_promedio = pliegues['Shape__Length'].mean()
    print(f"\n   Longitud total de pliegues: {longitud_total:,.2f} unidades")
    print(f"   Longitud promedio: {longitud_promedio:,.2f} unidades")

# 3. EXTRAER GEOMETRÍA DE SANTANDER
print("\n🗺️ Extrayendo Santander...")
santander_geoms = []
santander_bounds = {'lon_min': float('inf'), 'lon_max': float('-inf'),
                    'lat_min': float('inf'), 'lat_max': float('-inf')}

for feature in municipios['features']:
    if feature['properties'].get('DEPARTAMEN') == 'SANTANDER':
        try:
            geom = shape(feature['geometry'])
            santander_geoms.append(geom)
            
            # Calcular límites
            bounds = geom.bounds
            santander_bounds['lon_min'] = min(santander_bounds['lon_min'], bounds[0])
            santander_bounds['lat_min'] = min(santander_bounds['lat_min'], bounds[1])
            santander_bounds['lon_max'] = max(santander_bounds['lon_max'], bounds[2])
            santander_bounds['lat_max'] = max(santander_bounds['lat_max'], bounds[3])
        except:
            continue

print(f"   Municipios de Santander: {len(santander_geoms)}")

# 4. FILTRAR SISMOS DE SANTANDER
sismos_santander = []
for idx, sismo in sismos.iterrows():
    punto = Point(sismo['lon'], sismo['lat'])
    for geom in santander_geoms:
        if geom.contains(punto):
            sismos_santander.append(sismo)
            break

sismos_santander = pd.DataFrame(sismos_santander)
print(f"   Sismos en Santander: {len(sismos_santander)}")

# 5. IDENTIFICAR PLIEGUES EN SANTANDER
# (Los pliegues normalmente tienen coordenadas de geometría, vamos a estimarlo por nombre o región)
print("\n🔍 Identificando pliegues en Santander...")

# Si los pliegues tienen coordenadas en alguna columna
pliegues_cols = pliegues.columns.tolist()
print(f"   Columnas disponibles: {pliegues_cols}")

# Buscar columnas de coordenadas o región
pliegues_santander = []
if 'Region' in pliegues.columns or 'DEPARTAMEN' in pliegues.columns:
    # Filtrar por nombre de región
    for col in pliegues.columns:
        if 'region' in col.lower() or 'depart' in col.lower():
            pliegues_santander = pliegues[pliegues[col].str.contains('SANTANDER', case=False, na=False)]
            break

if len(pliegues_santander) == 0:
    print("   ⚠️ No se encontró columna de región, usando todos los pliegues")
    pliegues_santander = pliegues

print(f"   Pliegues identificados en zona: {len(pliegues_santander)}")

# ============================================
# VISUALIZACIÓN 1: COLOMBIA COMPLETA
# ============================================
print("\n🎨 Generando mapa de Colombia...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# --- PANEL 1: SISMOS Y PLIEGUES DE COLOMBIA ---
# Dibujar departamentos
for feature in municipios['features']:
    try:
        geom = shape(feature['geometry'])
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            poly = Polygon(list(zip(x, y)), 
                          facecolor='lightgray', 
                          edgecolor='black', 
                          linewidth=0.3, 
                          alpha=0.3)
            ax1.add_patch(poly)
        elif geom.geom_type == 'MultiPolygon':
            for poly_part in geom.geoms:
                x, y = poly_part.exterior.xy
                poly = Polygon(list(zip(x, y)), 
                              facecolor='lightgray', 
                              edgecolor='black', 
                              linewidth=0.3, 
                              alpha=0.3)
                ax1.add_patch(poly)
    except:
        continue

# Sismos (sample para no saturar)
sismos_muestra = sismos.sample(min(2000, len(sismos)))
ax1.scatter(sismos_muestra['lon'], sismos_muestra['lat'],
           c='red', s=1, alpha=0.3, label='Sismos')

# Resaltar Santander
for geom in santander_geoms:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        poly = Polygon(list(zip(x, y)), 
                      facecolor='yellow', 
                      edgecolor='red', 
                      linewidth=2, 
                      alpha=0.5)
        ax1.add_patch(poly)
    elif geom.geom_type == 'MultiPolygon':
        for poly_part in geom.geoms:
            x, y = poly_part.exterior.xy
            poly = Polygon(list(zip(x, y)), 
                          facecolor='yellow', 
                          edgecolor='red', 
                          linewidth=2, 
                          alpha=0.5)
            ax1.add_patch(poly)

ax1.set_xlim(-82, -66)
ax1.set_ylim(-5, 13)
ax1.set_aspect('equal')
ax1.set_xlabel('Longitud', fontsize=11)
ax1.set_ylabel('Latitud', fontsize=11)
ax1.set_title('Colombia: Actividad Sísmica y Pliegues Geológicos\n(Santander destacado en amarillo)', 
             fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- PANEL 2: DETALLE DE SANTANDER ---
# Dibujar límites de Santander
for geom in santander_geoms:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        poly = Polygon(list(zip(x, y)), 
                      facecolor='#f0f0f0', 
                      edgecolor='black', 
                      linewidth=1.5, 
                      alpha=0.5)
        ax2.add_patch(poly)
    elif geom.geom_type == 'MultiPolygon':
        for poly_part in geom.geoms:
            x, y = poly_part.exterior.xy
            poly = Polygon(list(zip(x, y)), 
                          facecolor='#f0f0f0', 
                          edgecolor='black', 
                          linewidth=1.5, 
                          alpha=0.5)
            ax2.add_patch(poly)

# Sismos de Santander
if len(sismos_santander) > 0:
    ax2.scatter(sismos_santander['lon'], sismos_santander['lat'],
               c='red', s=15, alpha=0.6, 
               edgecolors='black', linewidth=0.3,
               label=f'Sismos ({len(sismos_santander)})')

# Configurar límites
margen = 0.2
ax2.set_xlim(santander_bounds['lon_min'] - margen, 
             santander_bounds['lon_max'] + margen)
ax2.set_ylim(santander_bounds['lat_min'] - margen, 
             santander_bounds['lat_max'] + margen)
ax2.set_aspect('equal')
ax2.set_xlabel('Longitud', fontsize=11)
ax2.set_ylabel('Latitud', fontsize=11)
ax2.set_title(f'Santander: Detalle de Actividad Sísmica y Pliegues\n({len(sismos_santander)} sismos registrados)', 
             fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('colombia_pliegues_sismos.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: colombia_pliegues_sismos.png")

# ============================================
# ESTADÍSTICAS DETALLADAS
# ============================================
print("\n📊 ESTADÍSTICAS:")
print("=" * 60)
print(f"\n🇨🇴 COLOMBIA:")
print(f"   Total pliegues: {len(pliegues)}")
print(f"   Total sismos: {len(sismos)}")

if 'Tipo' in pliegues.columns:
    print(f"\n   Distribución de pliegues:")
    for tipo, cantidad in pliegues['Tipo'].value_counts().items():
        porcentaje = (cantidad / len(pliegues)) * 100
        print(f"      {tipo}: {cantidad} ({porcentaje:.1f}%)")

print(f"\n🏔️ SANTANDER:")
print(f"   Pliegues en zona: {len(pliegues_santander)}")
print(f"   Sismos: {len(sismos_santander)}")

if len(sismos_santander) > 0 and 'Mag.' in sismos_santander.columns:
    # Corregir magnitud de Santander
    sismos_santander['mag_corregida'] = sismos_santander['Mag.'] / 10000
    mag_promedio = sismos_santander['mag_corregida'].mean()
    mag_max = sismos_santander['mag_corregida'].max()
    print(f"   Magnitud promedio: {mag_promedio:.2f}")
    print(f"   Magnitud máxima: {mag_max:.2f}")

plt.show()

print("\n" + "=" * 60)
print("🎉 ¡ANÁLISIS COMPLETADO!")
print("=" * 60)