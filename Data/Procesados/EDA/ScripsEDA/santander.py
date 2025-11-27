import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
from shapely.geometry import Point, shape
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

print("🎯 ANÁLISIS DETALLADO: SANTANDER")
print("=" * 60)

# 1. CARGAR DATOS
print("\n📂 Cargando datos...")
sismos = pd.read_excel('Data/Procesados/LLCatálogo Sismicidad TECTO_limpio.xlsx')

with open('Data/Procesados/MunicipiosColombia.json', 'r', encoding='utf-8') as f:
    municipios = json.load(f)

# 2. CORREGIR COORDENADAS
sismos['lat'] = sismos['Lat(°)'] / 1000
sismos['lon'] = sismos['Long(°)'] / 1000
sismos['mag'] = sismos['Mag.'] / 10000  # Corregir magnitud también
sismos['prof'] = sismos['Prof(Km)'] / 1000  # Corregir profundidad

print(f"   Total sismos: {len(sismos)}")

# 3. EXTRAER GEOMETRÍA DE SANTANDER
print("\n🗺️ Extrayendo geometría de Santander...")
santander_geoms = []

for feature in municipios['features']:
    if feature['properties'].get('DEPARTAMEN') == 'SANTANDER':
        try:
            geom = shape(feature['geometry'])
            santander_geoms.append(geom)
        except:
            continue

print(f"   Municipios de Santander: {len(santander_geoms)}")

# 4. FILTRAR SISMOS EN SANTANDER
print("\n📍 Filtrando sismos de Santander...")
sismos_santander = []

for idx, sismo in sismos.iterrows():
    punto = Point(sismo['lon'], sismo['lat'])
    
    for geom in santander_geoms:
        if geom.contains(punto):
            sismos_santander.append(sismo)
            break

sismos_santander = pd.DataFrame(sismos_santander)
print(f"   Sismos en Santander: {len(sismos_santander)}")

# 5. ENCONTRAR ZONA DE MAYOR DENSIDAD
print("\n🔍 Identificando zona de mayor densidad...")

# Usar DBSCAN para encontrar clusters
coords = sismos_santander[['lon', 'lat']].values

# DBSCAN: eps en grados (~5km = 0.05 grados), min_samples=50
db = DBSCAN(eps=0.05, min_samples=50).fit(coords)
labels = db.labels_

# Encontrar el cluster más grande
unique_labels = set(labels)
unique_labels.discard(-1)  # Quitar ruido

cluster_sizes = {}
for label in unique_labels:
    cluster_sizes[label] = np.sum(labels == label)

if cluster_sizes:
    cluster_principal = max(cluster_sizes, key=cluster_sizes.get)
    mask_cluster = labels == cluster_principal
    
    # Centro del cluster (promedio de coordenadas)
    centro_lon = coords[mask_cluster, 0].mean()
    centro_lat = coords[mask_cluster, 1].mean()
    
    # Radio del cluster (distancia máxima desde el centro)
    distancias = np.sqrt(
        (coords[mask_cluster, 0] - centro_lon)**2 + 
        (coords[mask_cluster, 1] - centro_lat)**2
    )
    radio = np.percentile(distancias, 90)  # 90% de los puntos
    
    print(f"   ✅ Cluster principal encontrado:")
    print(f"      Centro: ({centro_lat:.4f}, {centro_lon:.4f})")
    print(f"      Sismos en cluster: {cluster_sizes[cluster_principal]}")
    print(f"      Radio: {radio:.4f}° (~{radio*111:.1f} km)")
else:
    # Si no hay clusters, usar el centroide general
    centro_lon = sismos_santander['lon'].mean()
    centro_lat = sismos_santander['lat'].mean()
    radio = 0.1

# 6. CREAR VISUALIZACIÓN
print("\n🎨 Generando visualización...")

fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# Dibujar límites de Santander
for geom in santander_geoms:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        poly = Polygon(list(zip(x, y)), 
                      facecolor='lightgray', 
                      edgecolor='black', 
                      linewidth=1.5, 
                      alpha=0.3)
        ax.add_patch(poly)
    elif geom.geom_type == 'MultiPolygon':
        for poly_part in geom.geoms:
            x, y = poly_part.exterior.xy
            poly = Polygon(list(zip(x, y)), 
                          facecolor='lightgray', 
                          edgecolor='black', 
                          linewidth=1.5, 
                          alpha=0.3)
            ax.add_patch(poly)

# Scatter de sismos con color por magnitud
scatter = ax.scatter(sismos_santander['lon'], 
                     sismos_santander['lat'],
                     c=sismos_santander['mag'],
                     s=sismos_santander['mag']*50 + 10,
                     cmap='YlOrRd',
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5,
                     vmin=0,
                     vmax=sismos_santander['mag'].max())

# Círculo de mayor densidad
circulo = Circle((centro_lon, centro_lat), 
                 radio, 
                 color='red', 
                 fill=False, 
                 linewidth=3, 
                 linestyle='--',
                 label=f'Zona de mayor densidad\n({cluster_sizes.get(cluster_principal, 0)} sismos)')
ax.add_patch(circulo)

# Marcar el centro
ax.plot(centro_lon, centro_lat, 'r*', 
        markersize=20, 
        markeredgecolor='black',
        markeredgewidth=1.5,
        label='Epicentro de densidad')

# Configuración de ejes
ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')
ax.set_title('Actividad Sísmica en Santander (2014-2023)\nZona de Mayor Densidad', 
             fontsize=16, fontweight='bold', pad=20)

# Barra de color
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Magnitud', rotation=270, labelpad=20, fontsize=12)

# Leyenda
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('santander_densidad_sismica.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: santander_densidad_sismica.png")

# 7. ESTADÍSTICAS DETALLADAS
print("\n📊 ESTADÍSTICAS DE SANTANDER:")
print("=" * 60)
print(f"Total de sismos: {len(sismos_santander)}")
print(f"Magnitud promedio: {sismos_santander['mag'].mean():.2f}")
print(f"Magnitud máxima: {sismos_santander['mag'].max():.2f}")
print(f"Profundidad promedio: {sismos_santander['prof'].mean():.1f} km")
print(f"Profundidad máxima: {sismos_santander['prof'].max():.1f} km")

# 8. MAPA DE DENSIDAD (HEATMAP)
print("\n🔥 Generando mapa de calor...")

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 12))

# Dibujar límites
for geom in santander_geoms:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        poly = Polygon(list(zip(x, y)), 
                      facecolor='none', 
                      edgecolor='black', 
                      linewidth=2)
        ax2.add_patch(poly)
    elif geom.geom_type == 'MultiPolygon':
        for poly_part in geom.geoms:
            x, y = poly_part.exterior.xy
            poly = Polygon(list(zip(x, y)), 
                          facecolor='none', 
                          edgecolor='black', 
                          linewidth=2)
            ax2.add_patch(poly)

# Hexbin (mapa de calor hexagonal)
hexbin = ax2.hexbin(sismos_santander['lon'], 
                     sismos_santander['lat'],
                     gridsize=30,
                     cmap='YlOrRd',
                     alpha=0.7,
                     edgecolors='black',
                     linewidths=0.2)

# Círculo de mayor densidad
circulo2 = Circle((centro_lon, centro_lat), 
                  radio, 
                  color='blue', 
                  fill=False, 
                  linewidth=3, 
                  linestyle='--')
ax2.add_patch(circulo2)

ax2.plot(centro_lon, centro_lat, 'b*', 
         markersize=20, 
         markeredgecolor='white',
         markeredgewidth=2)

ax2.set_xlabel('Longitud', fontsize=12, fontweight='bold')
ax2.set_ylabel('Latitud', fontsize=12, fontweight='bold')
ax2.set_title('Mapa de Calor - Densidad Sísmica en Santander\n(2014-2023)', 
              fontsize=16, fontweight='bold', pad=20)

cbar2 = plt.colorbar(hexbin, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Número de Sismos', rotation=270, labelpad=20, fontsize=12)

ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('santander_mapa_calor.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: santander_mapa_calor.png")

plt.show()

print("\n" + "=" * 60)
print("🎉 ¡ANÁLISIS DE SANTANDER COMPLETADO!")
print("=" * 60)