import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODELO DE PREDICCIÓN - AMENAZA SÍSMICA")
print("="*80)

# Rutas
archivo_sismos = r'Data\Procesados\LLCatálogo Sismicidad TECTO_limpio.xlsx'
carpeta_salida = 'Data/Procesados/MLFF'
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar datos ya limpios
print("\n[1/8] Cargando datos limpios...")
sismos = pd.read_excel(archivo_sismos)
print(f"Registros: {len(sismos)}")

# Identificar columnas
col_mag = 'Mag.'
col_region = 'Region'

# Agrupar por municipio
print("\n[2/8] Agrupando por municipio...")
sismos_agg = sismos.groupby(col_region).agg({
    col_mag: ['count', 'max', 'mean', 'std']
}).reset_index()

sismos_agg.columns = ['municipio', 'sismos_total', 'magnitud_max', 
                      'magnitud_media', 'magnitud_std']
sismos_agg['magnitud_std'] = sismos_agg['magnitud_std'].fillna(0)

print(f"Municipios: {len(sismos_agg)}")

# Crear features
print("\n[3/8] Creando features...")
sismos_agg['densidad'] = sismos_agg['sismos_total'] / sismos_agg['sismos_total'].max()
sismos_agg['peligrosidad'] = sismos_agg['magnitud_max'] * np.log1p(sismos_agg['sismos_total'])
sismos_agg['actividad_alta'] = (sismos_agg['sismos_total'] > sismos_agg['sismos_total'].quantile(0.75)).astype(int)
sismos_agg['variabilidad_alta'] = (sismos_agg['magnitud_std'] > sismos_agg['magnitud_std'].median()).astype(int)

# Definir amenaza
print("\n[4/8] Clasificando amenaza...")
def clasificar_amenaza(row):
    if row['magnitud_max'] >= 3.5:
        return 2  # ALTA
    elif row['magnitud_max'] >= 2.5 or row['actividad_alta'] == 1:
        return 1  # MEDIA
    else:
        return 0  # BAJA

sismos_agg['amenaza'] = sismos_agg.apply(clasificar_amenaza, axis=1)

# Distribución
clases = ['BAJA', 'MEDIA', 'ALTA']
for i, nombre in enumerate(clases):
    count = (sismos_agg['amenaza'] == i).sum()
    pct = count / len(sismos_agg) * 100
    print(f"  {nombre}: {count} ({pct:.1f}%)")

# Preparar datos
print("\n[5/8] Preparando entrenamiento...")
features = ['sismos_total', 'magnitud_max', 'magnitud_media', 'magnitud_std',
            'densidad', 'peligrosidad', 'actividad_alta', 'variabilidad_alta']

X = sismos_agg[features]
y = sismos_agg['amenaza']

# Split y escalado
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Entrenar modelos
print("\n[6/8] Entrenando modelos...")
modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                           class_weight='balanced', max_depth=10),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

resultados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train_sc, y_train)
    y_pred = modelo.predict(X_test_sc)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    resultados[nombre] = {
        'modelo': modelo,
        'accuracy': acc,
        'f1': f1,
        'y_pred': y_pred
    }
    print(f"  {nombre}: Acc={acc:.3f} | F1={f1:.3f}")

# Mejor modelo
print("\n[7/8] Seleccionando mejor...")
mejor = max(resultados.items(), key=lambda x: x[1]['f1'])
mejor_nombre = mejor[0]
mejor_modelo = mejor[1]['modelo']
y_pred = mejor[1]['y_pred']

print(f"\nMejor: {mejor_nombre}")
print(f"F1: {mejor[1]['f1']:.4f}")
print(f"Accuracy: {mejor[1]['accuracy']:.4f}")

print(f"\nReporte:")
print(classification_report(y_test, y_pred, target_names=clases, zero_division=0))

# Guardar
print("\n[8/8] Guardando...")
with open(os.path.join(carpeta_salida, 'modelo.pkl'), 'wb') as f:
    pickle.dump(mejor_modelo, f)

with open(os.path.join(carpeta_salida, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(carpeta_salida, 'features.txt'), 'w') as f:
    f.write('\n'.join(features))

# Comparación
comp_df = pd.DataFrame({
    'modelo': list(resultados.keys()),
    'accuracy': [r['accuracy'] for r in resultados.values()],
    'f1': [r['f1'] for r in resultados.values()]
}).sort_values('f1', ascending=False)

comp_df.to_csv(os.path.join(carpeta_salida, 'comparacion.csv'), index=False)

# Visualización
print("\nGenerando gráficas...")
fig = plt.figure(figsize=(15, 10))

# Comparación modelos
ax1 = plt.subplot(2, 3, 1)
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(comp_df))]
ax1.barh(comp_df['modelo'], comp_df['f1'], color=colors)
ax1.set_xlabel('F1-Score')
ax1.set_title('Comparación de Modelos')
for i, (idx, row) in enumerate(comp_df.iterrows()):
    ax1.text(row['f1'] + 0.01, i, f"{row['f1']:.3f}", va='center')

# Matriz confusión
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=clases, yticklabels=clases)
ax2.set_ylabel('Real')
ax2.set_xlabel('Predicción')
ax2.set_title(f'Matriz - {mejor_nombre}')

# Importancia features
ax3 = plt.subplot(2, 3, 3)
if hasattr(mejor_modelo, 'feature_importances_'):
    imp = pd.DataFrame({
        'feature': features,
        'imp': mejor_modelo.feature_importances_
    }).sort_values('imp', ascending=False)
    ax3.barh(imp['feature'].head(6), imp['imp'].head(6), color='#e74c3c')
    ax3.set_xlabel('Importancia')
    ax3.set_title('Top Features')
else:
    ax3.text(0.5, 0.5, 'N/A', ha='center', va='center')
    ax3.set_title('Importancia Features')
    ax3.axis('off')

# Distribución predicciones
ax4 = plt.subplot(2, 3, 4)
pred_dist = pd.Series(y_pred).value_counts().sort_index()
colors_pred = ['#2ecc71', '#f39c12', '#e74c3c']
ax4.bar(clases[:len(pred_dist)], pred_dist.values, color=colors_pred, alpha=0.7)
ax4.set_ylabel('Cantidad')
ax4.set_title('Predicciones')
ax4.grid(axis='y', alpha=0.3)

# Real vs Predicción
ax5 = plt.subplot(2, 3, 5)
real = pd.Series(y_test).value_counts().sort_index()
pred = pd.Series(y_pred).value_counts().sort_index()
x = np.arange(len(real))
w = 0.35
ax5.bar(x - w/2, real.values, w, label='Real', color='#3498db', alpha=0.7)
ax5.bar(x + w/2, pred.values, w, label='Pred', color='#e74c3c', alpha=0.7)
ax5.set_xticks(x)
ax5.set_xticklabels(clases[:len(x)])
ax5.set_ylabel('Cantidad')
ax5.set_title('Real vs Predicción')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Magnitud por amenaza
ax6 = plt.subplot(2, 3, 6)
for i in sorted(sismos_agg['amenaza'].unique()):
    datos = sismos_agg[sismos_agg['amenaza'] == i]['magnitud_max']
    ax6.hist(datos, bins=15, alpha=0.5, label=clases[i])
ax6.set_xlabel('Magnitud Máxima')
ax6.set_ylabel('Frecuencia')
ax6.set_title('Magnitud por Amenaza')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, 'evaluacion.png'), dpi=300, bbox_inches='tight')
print(f"Gráfico guardado: evaluacion.png")

# Resumen
print("\n" + "="*80)
print("COMPLETADO")
print("="*80)
print(f"\nArchivos en {carpeta_salida}:")
print("  - modelo.pkl")
print("  - scaler.pkl")
print("  - features.txt")
print("  - comparacion.csv")
print("  - evaluacion.png")
print(f"\nMejor modelo: {mejor_nombre}")
print(f"F1: {mejor[1]['f1']:.4f} | Acc: {mejor[1]['accuracy']:.4f}")
print(f"Municipios: {len(sismos_agg)} | Train: {len(X_train)} | Test: {len(X_test)}")
print("="*80)