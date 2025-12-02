import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            f1_score, cohen_kappa_score)
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("MODELO DE PREDICCION - AMENAZA SISMICA")

# Rutas
archivo_sismos = r'Data\Procesados\LLCatálogo Sismicidad TECTO_limpio.xlsx'
carpeta_salida = 'Data/Procesados/MLFF'
os.makedirs(carpeta_salida, exist_ok=True)

print("\n[1/9] Cargando datos...")
sismos = pd.read_excel(archivo_sismos)
print(f"Registros: {len(sismos)}")

print("\n[2/9] Agrupando por municipio...")
col_mag = 'Mag.'
col_region = 'Region'

sismos_agg = sismos.groupby(col_region).agg({
    col_mag: ['count', 'max', 'mean', 'std', 'median']
}).reset_index()

sismos_agg.columns = ['municipio', 'sismos_total', 'magnitud_max', 
                      'magnitud_media', 'magnitud_std', 'magnitud_mediana']
sismos_agg['magnitud_std'] = sismos_agg['magnitud_std'].fillna(0)

print(f"Municipios: {len(sismos_agg)}")

print("\n[3/9] Clasificando amenaza...")
def clasificar_amenaza_real(mag_max, sismos_count):
    if mag_max >= 3.5:
        return 2  # ALTA
    elif mag_max >= 2.5 or sismos_count > 50:
        return 1  # MEDIA
    else:
        return 0  # BAJA

sismos_agg['amenaza'] = sismos_agg.apply(
    lambda x: clasificar_amenaza_real(x['magnitud_max'], x['sismos_total']), 
    axis=1
)

clases = ['BAJA', 'MEDIA', 'ALTA']
dist_amenaza = []
for i, nombre in enumerate(clases):
    count = (sismos_agg['amenaza'] == i).sum()
    pct = count / len(sismos_agg) * 100
    dist_amenaza.append({'clase': nombre, 'count': count, 'pct': pct})
    print(f"  {nombre}: {count} ({pct:.1f}%)")


print("\n[4/9] Split estratificado...")
features_base = ['sismos_total', 'magnitud_media', 'magnitud_std', 'magnitud_mediana']

X = sismos_agg[features_base]
y = sismos_agg['amenaza']

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
except ValueError as e:
    print(f"⚠️  No se puede estratificar: {e}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print("\n[5/9] Creando features (sin leakage)...")

# estadIsticas SOLO del conjunto de entrenamiento
train_max = X_train['sismos_total'].max()
train_q75 = X_train['sismos_total'].quantile(0.75)
train_std_median = X_train['magnitud_std'].median()

# aplicar a train
X_train_features = X_train.copy()
X_train_features['densidad'] = X_train_features['sismos_total'] / train_max
X_train_features['actividad_alta'] = (X_train_features['sismos_total'] > train_q75).astype(int)
X_train_features['variabilidad_alta'] = (X_train_features['magnitud_std'] > train_std_median).astype(int)

# aplicar a test
X_test_features = X_test.copy()
X_test_features['densidad'] = X_test_features['sismos_total'] / train_max
X_test_features['actividad_alta'] = (X_test_features['sismos_total'] > train_q75).astype(int)
X_test_features['variabilidad_alta'] = (X_test_features['magnitud_std'] > train_std_median).astype(int)

features_finales = list(X_train_features.columns)

# escalado
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_features)
X_test_sc = scaler.transform(X_test_features)

print("\n[6/9] Entrenamiento de multiples modelos...")

modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        random_state=42, class_weight='balanced'
    ),
    'Gradient Boost': GradientBoostingClassifier(
        n_estimators=50, max_depth=3, min_samples_split=10,
        min_samples_leaf=5, learning_rate=0.05, subsample=0.8,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42, class_weight='balanced'
    ),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance')
}

resultados = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nombre, modelo in modelos.items():
    # validacion cruzada
    cv_scores = cross_val_score(modelo, X_train_sc, y_train, cv=cv, scoring='f1_weighted')
    
    # entrenar en todo el train
    modelo.fit(X_train_sc, y_train)
    y_pred = modelo.predict(X_test_sc)
    
    # metricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    
    resultados[nombre] = {
        'modelo': modelo,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'accuracy': acc,
        'f1': f1,
        'kappa': kappa,
        'y_pred': y_pred
    }
    
    print(f"  {nombre}:")
    print(f"    CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"    Test - Acc: {acc:.3f} | F1: {f1:.3f} | Kappa: {kappa:.3f}")

print("\n[7/9] seleccionando mejor modelo...")
mejor = max(resultados.items(), key=lambda x: x[1]['f1'])
mejor_nombre = mejor[0]
mejor_modelo = mejor[1]['modelo']
y_pred = mejor[1]['y_pred']

print(f"\n mejor modelo: {mejor_nombre}")
print(f"   F1 Test: {mejor[1]['f1']:.4f}")
print(f"   F1 CV: {mejor[1]['cv_mean']:.4f} ± {mejor[1]['cv_std']:.4f}")
print(f"   Kappa: {mejor[1]['kappa']:.4f}")

print(f"\n reporte de clasificacion:")
print(classification_report(y_test, y_pred, target_names=clases, zero_division=0))


print("\n[8/9] guardando modelos y metadata...")

with open(os.path.join(carpeta_salida, 'modelo.pkl'), 'wb') as f:
    pickle.dump(mejor_modelo, f)

with open(os.path.join(carpeta_salida, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(carpeta_salida, 'features.txt'), 'w') as f:
    f.write('\n'.join(features_finales))

# guardar estadisticas de train
stats_train = {
    'max_sismos': float(train_max),
    'q75_sismos': float(train_q75),
    'median_std': float(train_std_median)
}

with open(os.path.join(carpeta_salida, 'train_stats.pkl'), 'wb') as f:
    pickle.dump(stats_train, f)

# comparacion de modelos
comp_df = pd.DataFrame({
    'modelo': list(resultados.keys()),
    'f1_cv': [r['cv_mean'] for r in resultados.values()],
    'f1_test': [r['f1'] for r in resultados.values()],
    'accuracy': [r['accuracy'] for r in resultados.values()],
    'kappa': [r['kappa'] for r in resultados.values()]
}).sort_values('f1_test', ascending=False)

comp_df.to_csv(os.path.join(carpeta_salida, 'comparacion_modelos.csv'), index=False)

print("\n[9/9] generando visualizaciones...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# --- GRAFICO 1: Comparacion de Modelos (F1) ---
ax1 = fig.add_subplot(gs[0, :2])
colors_comp = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(comp_df))]
bars = ax1.barh(comp_df['modelo'], comp_df['f1_test'], color=colors_comp, alpha=0.8)
ax1.set_xlabel('F1-Score (Test)', fontsize=11, fontweight='bold')
ax1.set_title('Comparacion de modelos', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(comp_df.iterrows()):
    ax1.text(row['f1_test'] + 0.02, i, f"{row['f1_test']:.3f}", 
             va='center', fontsize=10, fontweight='bold')

# --- GRAFICO 2: matriz de Confusion ---
ax2 = fig.add_subplot(gs[0, 2:])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax2,
            xticklabels=clases, yticklabels=clases, cbar_kws={'label': 'Frecuencia'})
ax2.set_ylabel('Real', fontsize=11, fontweight='bold')
ax2.set_xlabel('Prediccion', fontsize=11, fontweight='bold')
ax2.set_title(f' Matriz de Confusion - {mejor_nombre}', fontsize=13, fontweight='bold', pad=15)

# --- GRAFICO 3: validación Cruzada (CV Scores) ---
ax3 = fig.add_subplot(gs[1, 0])
cv_data = []
for nombre, res in resultados.items():
    for score in res['cv_scores']:
        cv_data.append({'Modelo': nombre, 'F1': score})
cv_df = pd.DataFrame(cv_data)

sns.boxplot(data=cv_df, y='Modelo', x='F1', ax=ax3, palette='Set2')
ax3.set_xlabel('F1-Score', fontsize=10, fontweight='bold')
ax3.set_ylabel('')
ax3.set_title(' Validación Cruzada (5-Fold)', fontsize=12, fontweight='bold', pad=10)
ax3.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='Umbral 0.6')
ax3.legend(fontsize=8)
ax3.grid(axis='x', alpha=0.3)

# --- GRAFICO 4: importancia de Features ---
ax4 = fig.add_subplot(gs[1, 1])
if hasattr(mejor_modelo, 'feature_importances_'):
    imp = pd.DataFrame({
        'feature': features_finales,
        'importancia': mejor_modelo.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    colors_feat = plt.cm.Spectral(np.linspace(0, 1, len(imp)))
    ax4.barh(imp['feature'], imp['importancia'], color=colors_feat)
    ax4.set_xlabel('Importancia', fontsize=10, fontweight='bold')
    ax4.set_title('🔍 Importancia de Features', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(axis='x', alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No disponible\npara este modelo', 
             ha='center', va='center', fontsize=12, style='italic')
    ax4.set_title('🔍 Importancia de Features', fontsize=12, fontweight='bold', pad=10)
    ax4.axis('off')

# --- GRAFICO 5: distribucion de Clases (Real vs Predicho) ---
ax5 = fig.add_subplot(gs[1, 2])
x_pos = np.arange(len(clases))
width = 0.35

real_counts = [sum(y_test == i) for i in range(len(clases))]
pred_counts = [sum(y_pred == i) for i in range(len(clases))]

ax5.bar(x_pos - width/2, real_counts, width, label='Real', color='#3498db', alpha=0.8)
ax5.bar(x_pos + width/2, pred_counts, width, label='Predicho', color='#e74c3c', alpha=0.8)

ax5.set_ylabel('Cantidad', fontsize=10, fontweight='bold')
ax5.set_xlabel('Clase', fontsize=10, fontweight='bold')
ax5.set_title('📈 Real vs Predicción (Test)', fontsize=12, fontweight='bold', pad=10)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(clases)
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# --- GRAFICO 6: Metricas Comparativas ---
ax6 = fig.add_subplot(gs[1, 3])
metricas_names = ['F1\nCV', 'F1\nTest', 'Accuracy', 'Kappa']
metricas_vals = [
    mejor[1]['cv_mean'],
    mejor[1]['f1'],
    mejor[1]['accuracy'],
    mejor[1]['kappa']
]
colors_met = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']

bars_met = ax6.bar(metricas_names, metricas_vals, color=colors_met, alpha=0.8)
ax6.set_ylim(0, 1)
ax6.set_ylabel('Score', fontsize=10, fontweight='bold')
ax6.set_title(f'Métricas - {mejor_nombre}', fontsize=12, fontweight='bold', pad=10)
ax6.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excelente')
ax6.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Bueno')
ax6.legend(fontsize=8, loc='lower right')
ax6.grid(axis='y', alpha=0.3)

for bar, val in zip(bars_met, metricas_vals):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- GRAFICO 7: Distribucion de Magnitud por Amenaza ---
ax7 = fig.add_subplot(gs[2, :2])
for i in sorted(sismos_agg['amenaza'].unique()):
    datos = sismos_agg[sismos_agg['amenaza'] == i]['magnitud_max']
    ax7.hist(datos, bins=20, alpha=0.6, label=clases[i], edgecolor='black')

ax7.set_xlabel('Magnitud Máxima', fontsize=11, fontweight='bold')
ax7.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax7.set_title(' Distribución de Magnitud por Clase de Amenaza', fontsize=13, fontweight='bold', pad=15)
ax7.legend(fontsize=10)
ax7.grid(axis='y', alpha=0.3)

# --- GRAFICO 8: Accuracy por Clase ---
ax8 = fig.add_subplot(gs[2, 2])
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1_scores, support = precision_recall_fscore_support(
    y_test, y_pred, labels=[0, 1, 2], zero_division=0
)

x_clase = np.arange(len(clases))
width_clase = 0.25

ax8.bar(x_clase - width_clase, precision, width_clase, label='Precision', color='#3498db', alpha=0.8)
ax8.bar(x_clase, recall, width_clase, label='Recall', color='#2ecc71', alpha=0.8)
ax8.bar(x_clase + width_clase, f1_scores, width_clase, label='F1', color='#e74c3c', alpha=0.8)

ax8.set_ylabel('Score', fontsize=10, fontweight='bold')
ax8.set_xlabel('Clase', fontsize=10, fontweight='bold')
ax8.set_title(' Métricas por Clase', fontsize=12, fontweight='bold', pad=10)
ax8.set_xticks(x_clase)
ax8.set_xticklabels(clases)
ax8.set_ylim(0, 1.1)
ax8.legend(fontsize=9)
ax8.grid(axis='y', alpha=0.3)

# --- GRAFICO 9: Distribucion de Amenazas (Dataset Completo) ---
ax9 = fig.add_subplot(gs[2, 3])
dist_df = pd.DataFrame(dist_amenaza)
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
wedges, texts, autotexts = ax9.pie(
    dist_df['count'], 
    labels=dist_df['clase'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    explode=(0.05, 0.05, 0.1)
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax9.set_title('Distribución de Amenazas\n(Dataset Completo)', 
              fontsize=12, fontweight='bold', pad=10)

# titulo general
fig.suptitle('EVALUACIÓN COMPLETA DEL MODELO DE PREDICCIÓN SÍSMICA', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(os.path.join(carpeta_salida, 'evaluacion_completa.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f" Grafico guardado: evaluacion_completa.png")