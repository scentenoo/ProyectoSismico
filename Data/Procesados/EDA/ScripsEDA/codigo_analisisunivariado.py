import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

archivo_sismos = 'Data/Originals/Catálogo Sismicidad TECTO.csv'
archivo_pliegues = 'Data/Procesados/pliegues_limpios.csv'
carpeta_resultados = 'Data/Procesados/EDA'

os.makedirs(carpeta_resultados, exist_ok=True)

try:
    df_sismos = pd.read_csv(archivo_sismos)
    print(f"✅ Sismos cargados: {len(df_sismos)} registros")
except Exception as e:
    print(f"❌ Error cargando sismos: {e}")
    df_sismos = None

try:
    df_pliegues = pd.read_csv(archivo_pliegues)
    print(f"✅ Pliegues cargados: {len(df_pliegues)} registros")
except Exception as e:
    print(f"❌ Error cargando pliegues: {e}")
    df_pliegues = None

if df_sismos is not None:
    columna_mag = None
    for col in ['Mag.', 'Mag', 'Magnitud', 'magnitude', 'MAG']:
        if col in df_sismos.columns:
            columna_mag = col
            break
    
    if columna_mag:
        df_sismos[columna_mag] = pd.to_numeric(df_sismos[columna_mag], errors='coerce')
        magnitudes = df_sismos[columna_mag].dropna()
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribución de Sismos por Magnitud', fontsize=16, fontweight='bold')
        
        # 1. Histograma
        axes[0, 0].hist(magnitudes, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Magnitud', fontsize=12)
        axes[0, 0].set_ylabel('Frecuencia', fontsize=12)
        axes[0, 0].set_title('Histograma de Magnitudes', fontsize=14)
        axes[0, 0].axvline(magnitudes.mean(), color='red', linestyle='--', 
                           label=f'Media: {magnitudes.mean():.2f}')
        axes[0, 0].axvline(magnitudes.median(), color='green', linestyle='--', 
                           label=f'Mediana: {magnitudes.median():.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Boxplot
        axes[0, 1].boxplot(magnitudes, vert=True)
        axes[0, 1].set_ylabel('Magnitud', fontsize=12)
        axes[0, 1].set_title('Diagrama de Caja - Magnitudes', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribución por rangos
        rangos = [0, 2, 3, 4, 5, 10]
        etiquetas = ['<2.0', '2.0-3.0', '3.0-4.0', '4.0-5.0', '≥5.0']
        magnitudes_cat = pd.cut(magnitudes, bins=rangos, labels=etiquetas)
        conteo_rangos = magnitudes_cat.value_counts().sort_index()
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        axes[1, 0].bar(range(len(conteo_rangos)), conteo_rangos.values, 
                       color=colors, edgecolor='black')
        axes[1, 0].set_xticks(range(len(conteo_rangos)))
        axes[1, 0].set_xticklabels(etiquetas, rotation=0)
        axes[1, 0].set_xlabel('Rango de Magnitud', fontsize=12)
        axes[1, 0].set_ylabel('Cantidad de Sismos', fontsize=12)
        axes[1, 0].set_title('Sismos por Rango de Magnitud', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Agregar valores encima de las barras
        for i, v in enumerate(conteo_rangos.values):
            axes[1, 0].text(i, v + max(conteo_rangos.values)*0.01, str(v), 
                           ha='center', va='bottom', fontweight='bold')
        
        # 4. Gráfico de densidad
        magnitudes.plot(kind='density', ax=axes[1, 1], color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Magnitud', fontsize=12)
        axes[1, 1].set_ylabel('Densidad', fontsize=12)
        axes[1, 1].set_title('Densidad de Probabilidad - Magnitudes', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(magnitudes.mean(), color='red', linestyle='--', 
                          label='Media')
        axes[1, 1].legend()
        
        plt.tight_layout()
        archivo_mag = os.path.join(carpeta_resultados, '1_distribucion_magnitudes.png')
        plt.savefig(archivo_mag, dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfico guardado: {archivo_mag}")
        plt.close()

if df_pliegues is not None:
    # Identificar columna de tipo
    columna_tipo = None
    for col in ['tipo', 'Tipo', 'Type', 'tipo_pliegue', 'Tipo_Pliegue', 'TIPO']:
        if col in df_pliegues.columns:
            columna_tipo = col
            break
    
    if columna_tipo:
        tipos_pliegues = df_pliegues[columna_tipo].value_counts()
        
        print(f"\n📈 Estadísticas de Tipos de Pliegues:")
        print(f"   Total de pliegues: {len(df_pliegues)}")
        print(f"   Tipos diferentes: {len(tipos_pliegues)}")
        print(f"   Tipo más común: {tipos_pliegues.index[0]} ({tipos_pliegues.iloc[0]} pliegues)")
        
        # Crear figura con múltiples gráficos
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distribución de Tipos de Pliegues', fontsize=16, fontweight='bold')
        
        # 1. Gráfico de barras
        top_10 = tipos_pliegues.head(10)
        axes[0, 0].barh(range(len(top_10)), top_10.values, color='coral', edgecolor='black')
        axes[0, 0].set_yticks(range(len(top_10)))
        axes[0, 0].set_yticklabels(top_10.index)
        axes[0, 0].set_xlabel('Cantidad', fontsize=12)
        axes[0, 0].set_title('Top 10 Tipos de Pliegues', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Agregar valores
        for i, v in enumerate(top_10.values):
            axes[0, 0].text(v + max(top_10.values)*0.01, i, str(v), 
                           va='center', fontweight='bold')
        
        # 2. Gráfico de pastel (Top 5 + Otros)
        top_5 = tipos_pliegues.head(5)
        otros = tipos_pliegues.iloc[5:].sum()
        
        if otros > 0:
            labels = list(top_5.index) + ['Otros']
            sizes = list(top_5.values) + [otros]
        else:
            labels = list(top_5.index)
            sizes = list(top_5.values)
        
        colors_pie = plt.cm.Set3(range(len(sizes)))
        axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                       colors=colors_pie, textprops={'fontsize': 10})
        axes[0, 1].set_title('Proporción de Tipos de Pliegues (Top 5)', fontsize=14)
        
        # 3. Gráfico de barras completo (si hay muchos tipos)
        if len(tipos_pliegues) > 10:
            axes[1, 0].bar(range(len(tipos_pliegues)), tipos_pliegues.values, 
                          color='skyblue', edgecolor='black')
            axes[1, 0].set_xlabel('Tipo de Pliegue (ordenado)', fontsize=12)
            axes[1, 0].set_ylabel('Cantidad', fontsize=12)
            axes[1, 0].set_title(f'Todos los Tipos de Pliegues (n={len(tipos_pliegues)})', 
                                fontsize=14)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        else:
            axes[1, 0].bar(range(len(tipos_pliegues)), tipos_pliegues.values, 
                          color='skyblue', edgecolor='black')
            axes[1, 0].set_xticks(range(len(tipos_pliegues)))
            axes[1, 0].set_xticklabels(tipos_pliegues.index, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Cantidad', fontsize=12)
            axes[1, 0].set_title('Distribución de Tipos de Pliegues', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Tabla de frecuencias
        axes[1, 1].axis('off')
        tabla_data = []
        for tipo, count in tipos_pliegues.head(15).items():
            porcentaje = (count / len(df_pliegues)) * 100
            tabla_data.append([tipo, count, f'{porcentaje:.1f}%'])
        
        tabla = axes[1, 1].table(cellText=tabla_data,
                                colLabels=['Tipo', 'Cantidad', '%'],
                                cellLoc='left',
                                loc='center',
                                colWidths=[0.5, 0.25, 0.25])
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1, 2)
        axes[1, 1].set_title('Tabla de Frecuencias (Top 15)', fontsize=14, pad=20)
        
        plt.tight_layout()
        archivo_pliegues = os.path.join(carpeta_resultados, '2_distribucion_tipos_pliegues.png')
        plt.savefig(archivo_pliegues, dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfico guardado: {archivo_pliegues}")
        plt.close()

# ==================== 3. FRECUENCIA TEMPORAL DE SISMOS ====================
if df_sismos is not None:
    print("\n" + "="*70)
    print("📊 3. ANÁLISIS DE FRECUENCIA TEMPORAL DE SISMOS")
    print("="*70)
    
    # Identificar columna de fecha
    columna_fecha = None
    for col in df_sismos.columns:
        if 'fecha' in col.lower() or 'date' in col.lower():
            columna_fecha = col
            break
    
    if columna_fecha:
        # Convertir a datetime
        df_sismos[columna_fecha] = pd.to_datetime(df_sismos[columna_fecha], errors='coerce')
        df_temporal = df_sismos.dropna(subset=[columna_fecha])
        
        # Extraer componentes temporales
        df_temporal['año'] = df_temporal[columna_fecha].dt.year
        df_temporal['mes'] = df_temporal[columna_fecha].dt.month
        df_temporal['dia_semana'] = df_temporal[columna_fecha].dt.day_name()
        df_temporal['hora'] = df_temporal[columna_fecha].dt.hour
        
        print(f"\n📈 Estadísticas Temporales:")
        print(f"   Fecha más antigua: {df_temporal[columna_fecha].min()}")
        print(f"   Fecha más reciente: {df_temporal[columna_fecha].max()}")
        print(f"   Rango temporal: {(df_temporal[columna_fecha].max() - df_temporal[columna_fecha].min()).days} días")
        
        # Crear figura con múltiples gráficos
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Frecuencia Temporal de Sismos', fontsize=16, fontweight='bold')
        
        # 1. Serie temporal por año
        sismos_por_año = df_temporal['año'].value_counts().sort_index()
        axes[0, 0].plot(sismos_por_año.index, sismos_por_año.values, 
                       marker='o', linewidth=2, markersize=6, color='darkblue')
        axes[0, 0].fill_between(sismos_por_año.index, sismos_por_año.values, alpha=0.3)
        axes[0, 0].set_xlabel('Año', fontsize=12)
        axes[0, 0].set_ylabel('Número de Sismos', fontsize=12)
        axes[0, 0].set_title('Sismos por Año', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Distribución por mes
        sismos_por_mes = df_temporal['mes'].value_counts().sort_index()
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        axes[0, 1].bar(sismos_por_mes.index, sismos_por_mes.values, 
                      color='teal', edgecolor='black')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(meses, rotation=45)
        axes[0, 1].set_xlabel('Mes', fontsize=12)
        axes[0, 1].set_ylabel('Número de Sismos', fontsize=12)
        axes[0, 1].set_title('Distribución de Sismos por Mes', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Distribución por día de la semana
        orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_español = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        sismos_por_dia = df_temporal['dia_semana'].value_counts()
        sismos_ordenados = [sismos_por_dia.get(dia, 0) for dia in orden_dias]
        
        axes[1, 0].bar(range(7), sismos_ordenados, color='orange', edgecolor='black')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels(dias_español, rotation=45)
        axes[1, 0].set_xlabel('Día de la Semana', fontsize=12)
        axes[1, 0].set_ylabel('Número de Sismos', fontsize=12)
        axes[1, 0].set_title('Distribución de Sismos por Día de la Semana', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Distribución por hora del día
        sismos_por_hora = df_temporal['hora'].value_counts().sort_index()
        axes[1, 1].bar(sismos_por_hora.index, sismos_por_hora.values, 
                      color='crimson', edgecolor='black')
        axes[1, 1].set_xlabel('Hora del Día (UTC)', fontsize=12)
        axes[1, 1].set_ylabel('Número de Sismos', fontsize=12)
        axes[1, 1].set_title('Distribución de Sismos por Hora', fontsize=14)
        axes[1, 1].set_xticks(range(0, 24, 2))
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        archivo_temporal = os.path.join(carpeta_resultados, '3_frecuencia_temporal_sismos.png')
        plt.savefig(archivo_temporal, dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfico guardado: {archivo_temporal}")
        plt.close()

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print("✅ ANÁLISIS UNIVARIADO COMPLETADO")
print("="*70)
print(f"\n📁 Todos los gráficos guardados en: {carpeta_resultados}")
print("\nArchivos generados:")
print("   1️⃣  1_distribucion_magnitudes.png")
print("   2️⃣  2_distribucion_tipos_pliegues.png")
print("   3️⃣  3_frecuencia_temporal_sismos.png")
print("\n🎉 ¡Análisis exploratorio completado con éxito!")