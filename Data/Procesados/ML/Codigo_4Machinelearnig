import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("🔷 FASE 4: PREPARACIÓN PARA MACHINE LEARNING")
print("="*70)

# ==================== CONFIGURACIÓN ====================
archivo_sismos = 'Data/Originals/Catálogo Sismicidad TECTO.csv'
archivo_pliegues = 'Data/Procesados/pliegues_limpios.csv'
carpeta_resultados = 'Data/Procesados/ML'

# Crear carpeta
os.makedirs(carpeta_resultados, exist_ok=True)

# ==================== 4.1 CARGAR Y PREPARAR DATOS ====================
print("\n📥 Paso 1: Cargando datos...")

df_sismos = pd.read_csv(archivo_sismos)
print(f"✅ Sismos cargados: {len(df_sismos)} registros")

# ==================== IDENTIFICAR COLUMNAS ====================
print("\n🔍 Identificando columnas clave...")

columnas = {}
for col in df_sismos.columns:
    if 'mag' in col.lower() and 'tipo' not in col.lower():
        columnas['magnitud'] = col
    if 'prof' in col.lower():
        columnas['profundidad'] = col
    if 'lat' in col.lower():
        columnas['latitud'] = col
    if 'lon' in col.lower():
        columnas['longitud'] = col
    if 'region' in col.lower():
        columnas['region'] = col
    if 'gap' in col.lower():
        columnas['gap'] = col
    if 'rms' in col.lower():
        columnas['rms'] = col

print(f"   Columnas identificadas: {columnas}")

# ==================== LIMPIAR DATOS ====================
print("\n🧹 Limpiando datos...")

# Convertir a numérico
for key, col in columnas.items():
    if key != 'region':
        df_sismos[col] = pd.to_numeric(df_sismos[col], errors='coerce')

# Eliminar filas con valores nulos en columnas críticas
columnas_criticas = [columnas.get('magnitud'), columnas.get('profundidad'), 
                     columnas.get('latitud'), columnas.get('longitud')]
columnas_criticas = [c for c in columnas_criticas if c is not None]

df_limpio = df_sismos.dropna(subset=columnas_criticas)
print(f"   Datos limpios: {len(df_limpio)} registros")

# ==================== 4.1 DEFINICIÓN DEL TARGET ====================
print("\n" + "="*70)
print("🎯 4.1 DEFINICIÓN DEL TARGET")
print("="*70)

# Definir amenaza basada en magnitud
# Alta amenaza: Magnitud >= 4.0
# Baja/Media amenaza: Magnitud < 4.0

umbral_magnitud = 4.0
col_mag = columnas.get('magnitud')

if col_mag:
    df_limpio['target'] = (df_limpio[col_mag] >= umbral_magnitud).astype(int)
    
    print(f"\n✅ Target creado:")
    print(f"   Criterio: Magnitud >= {umbral_magnitud}")
    print(f"   1 = Alta amenaza (Magnitud >= {umbral_magnitud})")
    print(f"   0 = Baja/Media amenaza (Magnitud < {umbral_magnitud})")
    
    # Contar clases
    conteo_clases = df_limpio['target'].value_counts()
    print(f"\n📊 Distribución de clases:")
    print(f"   Clase 0 (Baja/Media): {conteo_clases.get(0, 0)} ({conteo_clases.get(0, 0)/len(df_limpio)*100:.1f}%)")
    print(f"   Clase 1 (Alta): {conteo_clases.get(1, 0)} ({conteo_clases.get(1, 0)/len(df_limpio)*100:.1f}%)")
else:
    print("❌ No se pudo crear el target (columna de magnitud no encontrada)")
    exit()

# ==================== 4.2 PREPROCESAMIENTO ====================
print("\n" + "="*70)
print("⚙️ 4.2 PREPROCESAMIENTO")
print("="*70)

# ==================== SELECCIÓN DE FEATURES ====================
print("\n📋 Seleccionando features...")

# Features numéricas
features_numericas = []
if columnas.get('profundidad'):
    features_numericas.append(columnas['profundidad'])
if columnas.get('latitud'):
    features_numericas.append(columnas['latitud'])
if columnas.get('longitud'):
    features_numericas.append(columnas['longitud'])
if columnas.get('gap'):
    features_numericas.append(columnas['gap'])
if columnas.get('rms'):
    features_numericas.append(columnas['rms'])

# Feature categórica
feature_categorica = columnas.get('region')

print(f"   Features numéricas: {features_numericas}")
print(f"   Feature categórica: {feature_categorica}")

# Crear DataFrame con features
df_features = df_limpio[features_numericas + ([feature_categorica] if feature_categorica else [])].copy()
df_features['target'] = df_limpio['target']

# Eliminar filas con NaN en features numéricas
df_features = df_features.dropna(subset=features_numericas)
print(f"\n   Registros después de limpieza: {len(df_features)}")

# ==================== ENCODING DE VARIABLES CATEGÓRICAS ====================
print("\n🔤 Encoding de variables categóricas...")

if feature_categorica and feature_categorica in df_features.columns:
    # Label Encoding para región
    le = LabelEncoder()
    df_features[f'{feature_categorica}_encoded'] = le.fit_transform(df_features[feature_categorica].astype(str))
    
    print(f"   ✅ Región codificada: {len(le.classes_)} categorías únicas")
    print(f"   Ejemplos: {list(le.classes_[:5])}")
    
    # Guardar el encoder
    import pickle
    with open(os.path.join(carpeta_resultados, 'label_encoder_region.pkl'), 'wb') as f:
        pickle.dump(le, f)
    print(f"   💾 Encoder guardado")
    
    # Agregar feature encoded a la lista
    features_numericas.append(f'{feature_categorica}_encoded')

# ==================== PREPARAR X Y y ====================
X = df_features[features_numericas].values
y = df_features['target'].values

print(f"\n📊 Dimensiones de los datos:")
print(f"   X (features): {X.shape}")
print(f"   y (target): {y.shape}")

# ==================== SPLIT TRAIN/TEST ====================
print("\n✂️ Split Train/Test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Train: {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   ✅ Test: {X_test.shape[0]} registros ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verificar distribución en train/test
print(f"\n   Distribución Train:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"      Clase {u}: {c} ({c/len(y_train)*100:.1f}%)")

print(f"   Distribución Test:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"      Clase {u}: {c} ({c/len(y_test)*100:.1f}%)")

# ==================== ESCALADO DE FEATURES ====================
print("\n📏 Escalado de features numéricas (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✅ Features escaladas")
print(f"   Media antes: {X_train[:, 0].mean():.2f}")
print(f"   Media después: {X_train_scaled[:, 0].mean():.2f}")
print(f"   Std después: {X_train_scaled[:, 0].std():.2f}")

# Guardar scaler
with open(os.path.join(carpeta_resultados, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"   💾 Scaler guardado")

# ==================== BALANCEO DE CLASES ====================
print("\n⚖️ Análisis de balanceo de clases...")

ratio_clases = np.bincount(y_train)[1] / np.bincount(y_train)[0]
print(f"   Ratio actual (Clase 1 / Clase 0): {ratio_clases:.3f}")

if ratio_clases < 0.5 or ratio_clases > 2.0:
    print(f"   ⚠️ Clases desbalanceadas detectadas")
    print(f"   Aplicando balanceo...")
    
    # Combinar X_train_scaled y y_train
    df_train = pd.DataFrame(X_train_scaled)
    df_train['target'] = y_train
    
    # Separar por clase
    df_mayoria = df_train[df_train['target'] == 0]
    df_minoria = df_train[df_train['target'] == 1]
    
    # Oversample de la clase minoritaria
    df_minoria_upsampled = resample(df_minoria,
                                    replace=True,
                                    n_samples=len(df_mayoria),
                                    random_state=42)
    
    # Combinar
    df_train_balanced = pd.concat([df_mayoria, df_minoria_upsampled])
    
    # Shuffle
    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42)
    
    # Separar features y target
    X_train_balanced = df_train_balanced.drop('target', axis=1).values
    y_train_balanced = df_train_balanced['target'].values
    
    print(f"   ✅ Balanceo aplicado:")
    print(f"      Train balanceado: {X_train_balanced.shape[0]} registros")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"      Clase {u}: {c} ({c/len(y_train_balanced)*100:.1f}%)")
    
    # Usar datos balanceados
    X_train_final = X_train_balanced
    y_train_final = y_train_balanced
else:
    print(f"   ✅ Clases relativamente balanceadas, no se requiere balanceo")
    X_train_final = X_train_scaled
    y_train_final = y_train

# ==================== GUARDAR DATOS PROCESADOS ====================
print("\n💾 Guardando datos procesados...")

# Guardar como numpy arrays
np.save(os.path.join(carpeta_resultados, 'X_train.npy'), X_train_final)
np.save(os.path.join(carpeta_resultados, 'y_train.npy'), y_train_final)
np.save(os.path.join(carpeta_resultados, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(carpeta_resultados, 'y_test.npy'), y_test)

print(f"   ✅ X_train.npy guardado ({X_train_final.shape})")
print(f"   ✅ y_train.npy guardado ({y_train_final.shape})")
print(f"   ✅ X_test.npy guardado ({X_test_scaled.shape})")
print(f"   ✅ y_test.npy guardado ({y_test.shape})")

# Guardar nombres de features
features_info = {
    'feature_names': features_numericas,
    'n_features': len(features_numericas)
}

import json
with open(os.path.join(carpeta_resultados, 'features_info.json'), 'w') as f:
    json.dump(features_info, f, indent=4)

print(f"   ✅ features_info.json guardado")

# ==================== VISUALIZACIÓN ====================
print("\n📊 Generando visualizaciones...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Preparación de Datos para Machine Learning', fontsize=16, fontweight='bold')

# 1. Distribución de clases original vs balanceada
ax1 = axes[0, 0]
clases = ['Baja/Media\n(0)', 'Alta\n(1)']
original = [np.sum(y_train == 0), np.sum(y_train == 1)]
balanceada = [np.sum(y_train_final == 0), np.sum(y_train_final == 1)]

x = np.arange(len(clases))
width = 0.35

ax1.bar(x - width/2, original, width, label='Original', color='steelblue')
ax1.bar(x + width/2, balanceada, width, label='Balanceada', color='coral')
ax1.set_xlabel('Clase', fontweight='bold')
ax1.set_ylabel('Cantidad', fontweight='bold')
ax1.set_title('Distribución de Clases')
ax1.set_xticks(x)
ax1.set_xticklabels(clases)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Distribución Train/Test
ax2 = axes[0, 1]
sizes = [len(X_train_final), len(X_test_scaled)]
labels_split = [f'Train\n({len(X_train_final)} - 80%)', f'Test\n({len(X_test_scaled)} - 20%)']
colors_split = ['#66b3ff', '#ff9999']
ax2.pie(sizes, labels=labels_split, autopct='%1.1f%%', startangle=90, colors=colors_split)
ax2.set_title('Split Train/Test')

# 3. Comparación antes/después de escalado
ax3 = axes[1, 0]
feature_idx = 0  # Primera feature
ax3.hist(X_train[:, feature_idx], bins=50, alpha=0.6, label='Antes de escalar', color='blue')
ax3.hist(X_train_scaled[:, feature_idx], bins=50, alpha=0.6, label='Después de escalar', color='red')
ax3.set_xlabel('Valor', fontweight='bold')
ax3.set_ylabel('Frecuencia', fontweight='bold')
ax3.set_title(f'Efecto del Escalado\n(Feature: {features_numericas[feature_idx]})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Tabla de resumen
ax4 = axes[1, 1]
ax4.axis('off')

tabla_data = [
    ['Métrica', 'Valor'],
    ['', ''],
    ['Total registros', f'{len(df_features)}'],
    ['Features numéricas', f'{len(features_numericas)}'],
    ['Train (80%)', f'{len(X_train_final)}'],
    ['Test (20%)', f'{len(X_test_scaled)}'],
    ['', ''],
    ['Umbral magnitud', f'{umbral_magnitud}'],
    ['Clase 0 (Train)', f'{np.sum(y_train_final==0)}'],
    ['Clase 1 (Train)', f'{np.sum(y_train_final==1)}'],
    ['', ''],
    ['Scaler', 'StandardScaler'],
    ['Balanceo', 'Sí' if ratio_clases < 0.5 or ratio_clases > 2.0 else 'No'],
]

tabla = ax4.table(cellText=tabla_data, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 2)
ax4.set_title('Resumen de Preparación', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
archivo_viz = os.path.join(carpeta_resultados, 'preparacion_ml_visualizacion.png')
plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
print(f"✅ Visualización guardada: {archivo_viz}")
plt.close()

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print("✅ PREPARACIÓN PARA ML COMPLETADA")
print("="*70)

print(f"\n📁 Archivos guardados en: {carpeta_resultados}")
print("   📄 X_train.npy - Features de entrenamiento")
print("   📄 y_train.npy - Target de entrenamiento")
print("   📄 X_test.npy - Features de prueba")
print("   📄 y_test.npy - Target de prueba")
print("   📄 scaler.pkl - Escalador entrenado")
print("   📄 label_encoder_region.pkl - Encoder de región")
print("   📄 features_info.json - Información de features")
print("   📊 preparacion_ml_visualizacion.png")

print("\n📊 Resumen:")
print(f"   ✅ Features: {len(features_numericas)}")
print(f"   ✅ Train: {X_train_final.shape}")
print(f"   ✅ Test: {X_test_scaled.shape}")
print(f"   ✅ Target: Magnitud >= {umbral_magnitud}")
print(f"   ✅ Escalado: StandardScaler")
print(f"   ✅ Balanceo: {'Aplicado' if ratio_clases < 0.5 or ratio_clases > 2.0 else 'No necesario'}")

print("\n🎉 ¡Datos listos para entrenar modelos de ML!")