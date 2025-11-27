import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from shapely.geometry import Point, shape, MultiPolygon
from shapely.prepared import prep
from shapely.ops import unary_union

print("🚀 INICIANDO ANÁLISIS DE SISMOS POR DEPARTAMENTO")
print("=" * 60)

# 1. CARGAR DATOS
print("\n📂 Cargando datos...")
try:
    sismos = pd.read_excel('Data/Procesados/LLCatálogo Sismicidad TECTO_limpio.xlsx')
    print(f"   ✅ Sismos cargados: {len(sismos)}")
except Exception as e:
    print(f"   ❌ Error cargando sismos: {e}")
    exit(1)

try:
    with open('Data/Procesados/MunicipiosColombia.json', 'r', encoding='utf-8') as f:
        municipios = json.load(f)
    print(f"   ✅ Municipios cargados: {len(municipios['features'])}")
except Exception as e:
    print(f"   ❌ Error cargando municipios: {e}")
    exit(1)

# 2. CORREGIR COORDENADAS
print("\n🔧 Corrigiendo coordenadas...")
sismos['lat'] = sismos['Lat(°)'] / 1000
sismos['lon'] = sismos['Long(°)'] / 1000

print(f"   Rango Lat: [{sismos['lat'].min():.2f}, {sismos['lat'].max():.2f}]")
print(f"   Rango Lon: [{sismos['lon'].min():.2f}, {sismos['lon'].max():.2f}]")

# 3. AGRUPAR POR DEPARTAMENTO
print("\n🗺️ Agrupando geometrías por departamento...")
departamentos_dict = {}

for idx, feature in enumerate(municipios['features']):
    try:
        dept_nombre = feature['properties'].get('DEPARTAMEN', '')
        
        if not dept_nombre:
            continue
        
        if dept_nombre not in departamentos_dict:
            departamentos_dict[dept_nombre] = {
                'geometrias': [],
                'sismos': 0
            }
        
        geom = shape(feature['geometry'])
        departamentos_dict[dept_nombre]['geometrias'].append(geom)
        
    except Exception as e:
        continue

print(f"   ✅ Departamentos encontrados: {len(departamentos_dict)}")

# 4. UNIR GEOMETRÍAS (CLAVE PARA VELOCIDAD)
print("\n⚡ Optimizando geometrías...")
for dept_nombre in list(departamentos_dict.keys()):
    try:
        geoms = departamentos_dict[dept_nombre]['geometrias']
        
        if len(geoms) > 0:
            # Unir todas las geometrías en una sola
            geom_unida = unary_union(geoms)
            departamentos_dict[dept_nombre]['geometria'] = geom_unida
            departamentos_dict[dept_nombre]['preparada'] = prep(geom_unida)
            print(f"   ✓ {dept_nombre}")
        else:
            del departamentos_dict[dept_nombre]
            
    except Exception as e:
        print(f"   ✗ Error en {dept_nombre}: {e}")
        del departamentos_dict[dept_nombre]

print(f"\n   ✅ {len(departamentos_dict)} departamentos listos")

# 5. ASIGNAR SISMOS (RÁPIDO CON GEOMETRÍAS PREPARADAS)
print("\n📍 Asignando sismos a departamentos...")
total = len(sismos)
sismos_asignados = 0

for idx, sismo in sismos.iterrows():
    if idx % 500 == 0:
        print(f"   Progreso: {idx}/{total} ({idx/total*100:.1f}%)")
    
    try:
        punto = Point(sismo['lon'], sismo['lat'])
        
        for dept_nombre, dept_data in departamentos_dict.items():
            if dept_data['preparada'].contains(punto):
                dept_data['sismos'] += 1
                sismos_asignados += 1
                break
    except:
        continue

print(f"\n   ✅ Sismos asignados: {sismos_asignados}/{total}")

# 6. RESULTADOS
print("\n📊 TOP 15 DEPARTAMENTOS:")
print("=" * 60)
sismos_dept = [(nombre, data['sismos']) for nombre, data in departamentos_dict.items()]
sismos_dept.sort(key=lambda x: x[1], reverse=True)

for i, (nombre, cantidad) in enumerate(sismos_dept[:15], 1):
    print(f"   {i:2d}. {nombre:25s}: {cantidad:5d} sismos")

# 7. MAPA DE CALOR
print("\n🎨 Generando mapa de calor...")

fig, ax = plt.subplots(1, 1, figsize=(12, 14))

max_sismos = max([data['sismos'] for data in departamentos_dict.values()])
cmap = plt.cm.YlOrRd

for dept_nombre, dept_data in departamentos_dict.items():
    if 'geometria' not in dept_data:
        continue
    
    n_sismos = dept_data['sismos']
    intensidad = n_sismos / max_sismos if max_sismos > 0 else 0
    color = cmap(intensidad)
    
    geom = dept_data['geometria']
    
    try:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            poly = Polygon(list(zip(x, y)), 
                          facecolor=color, 
                          edgecolor='black', 
                          linewidth=0.5, 
                          alpha=0.8)
            ax.add_patch(poly)
        
        elif geom.geom_type == 'MultiPolygon':
            for poly_part in geom.geoms:
                x, y = poly_part.exterior.xy
                poly = Polygon(list(zip(x, y)), 
                              facecolor=color, 
                              edgecolor='black', 
                              linewidth=0.5, 
                              alpha=0.8)
                ax.add_patch(poly)
    except:
        continue

ax.set_xlim(-82, -66)
ax.set_ylim(-5, 13)
ax.set_aspect('equal')
ax.set_xlabel('Longitud', fontsize=12)
ax.set_ylabel('Latitud', fontsize=12)
ax.set_title('Densidad de Sismos por Departamento\nColombia (2014-2023)', 
             fontsize=16, fontweight='bold', pad=20)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_sismos))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Número de Sismos', rotation=270, labelpad=20, fontsize=12)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('mapa_sismos_colombia.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: mapa_sismos_colombia.png")

# 8. GRÁFICO DE BARRAS
print("\n📊 Generando gráfico de barras...")

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))

top_15 = sismos_dept[:15]
nombres = [d[0] for d in top_15]
valores = [d[1] for d in top_15]
colores = [cmap(v/max_sismos) for v in valores]

bars = ax2.barh(nombres, valores, color=colores, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Número de Sismos', fontsize=12)
ax2.set_title('Top 15 Departamentos con Mayor Actividad Sísmica\n(2014-2023)', 
              fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

for i, (bar, valor) in enumerate(zip(bars, valores)):
    ax2.text(valor + max_sismos*0.01, i, f'{valor}', 
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('top_departamentos_sismos.png', dpi=300, bbox_inches='tight')
print("   ✅ Guardado: top_departamentos_sismos.png")

plt.show()

print("\n" + "=" * 60)
print("🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
print("=" * 60)