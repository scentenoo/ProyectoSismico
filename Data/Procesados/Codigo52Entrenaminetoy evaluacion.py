import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔧 FASE 5.2: ENTRENAMIENTO Y EVALUACIÓN AVANZADA")
print("   • Cross-validation (5-folds)")
print("   • Hyperparameter tuning")
print("   • Comparación de modelos")
print("   • Selección del mejor modelo")
print("="*70)

# ==================== CONFIGURACIÓN ====================
carpeta_ml = 'Data/Procesados/ML'
carpeta_modelos = 'Data/Procesados/ML/Modelos'
carpeta_optimizacion = 'Data/Procesados/ML/Optimizacion'
os.makedirs(carpeta_optimizacion, exist_ok=True)

# ==================== CARGAR DATOS ====================
print("\n📥 Cargando datos preprocesados...")

try:
    X_train = np.load(os.path.join(carpeta_ml, 'X_train.npy'))
    y_train = np.load(os.path.join(carpeta_ml, 'y_train.npy'))
    X_test = np.load(os.path.join(carpeta_ml, 'X_test.npy'))
    y_test = np.load(os.path.join(carpeta_ml, 'y_test.npy'))
    
    print(f"✅ Datos cargados: Train={X_train.shape}, Test={X_test.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ==================== CROSS-VALIDATION ====================
print("\n" + "="*70)
print("📊 PASO 1: CROSS-VALIDATION (5-FOLDS)")
print("="*70)

# Configurar cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'accuracy': 'accuracy', 
           'f1': make_scorer(f1_score),
           'precision': 'precision',
           'recall': 'recall'}

# Modelos base
modelos_base = {
    'K-NN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

resultados_cv = {}

print("\n🔄 Ejecutando cross-validation para cada modelo...")

for nombre, modelo in modelos_base.items():
    print(f"\n   📌 {nombre}...")
    inicio = time.time()
    
    # Cross-validation con múltiples métricas
    scores_acc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    scores_f1 = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    scores_prec = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='precision', n_jobs=-1)
    scores_rec = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1)
    
    tiempo = time.time() - inicio
    
    resultados_cv[nombre] = {
        'accuracy_mean': scores_acc.mean(),
        'accuracy_std': scores_acc.std(),
        'f1_mean': scores_f1.mean(),
        'f1_std': scores_f1.std(),
        'precision_mean': scores_prec.mean(),
        'precision_std': scores_prec.std(),
        'recall_mean': scores_rec.mean(),
        'recall_std': scores_rec.std(),
        'tiempo': tiempo
    }
    
    print(f"      ✅ Accuracy: {scores_acc.mean():.4f} (±{scores_acc.std():.4f})")
    print(f"      ✅ F1-Score: {scores_f1.mean():.4f} (±{scores_f1.std():.4f})")
    print(f"      ⏱️  Tiempo: {tiempo:.2f}s")

# Guardar resultados CV
df_cv = pd.DataFrame(resultados_cv).T
archivo_cv = os.path.join(carpeta_optimizacion, 'resultados_cross_validation.csv')
df_cv.to_csv(archivo_cv)
print(f"\n💾 Resultados CV guardados: {archivo_cv}")

# ==================== HYPERPARAMETER TUNING ====================
print("\n" + "="*70)
print("🎛️  PASO 2: HYPERPARAMETER TUNING")
print("="*70)

# Definir grids de hiperparámetros
param_grids = {
    'K-NN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf', 'linear']
    }
}

mejores_modelos = {}
resultados_tuning = {}

print("\n🔍 Buscando mejores hiperparámetros para cada modelo...")

for nombre, modelo in modelos_base.items():
    print(f"\n   🎯 Optimizando {nombre}...")
    inicio = time.time()
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grids[nombre],
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    tiempo = time.time() - inicio
    
    mejores_modelos[nombre] = grid_search.best_estimator_
    
    resultados_tuning[nombre] = {
        'mejor_score': grid_search.best_score_,
        'mejores_params': grid_search.best_params_,
        'tiempo': tiempo
    }
    
    print(f"      ✅ Mejor F1-Score (CV): {grid_search.best_score_:.4f}")
    print(f"      ✅ Mejores parámetros: {grid_search.best_params_}")
    print(f"      ⏱️  Tiempo: {tiempo:.2f}s")

# Guardar mejores parámetros
import json
archivo_params = os.path.join(carpeta_optimizacion, 'mejores_hiperparametros.json')
params_serializables = {}
for nombre, info in resultados_tuning.items():
    params_serializables[nombre] = {
        'mejor_score': float(info['mejor_score']),
        'mejores_params': {k: str(v) for k, v in info['mejores_params'].items()},
        'tiempo': float(info['tiempo'])
    }

with open(archivo_params, 'w') as f:
    json.dump(params_serializables, f, indent=4)
print(f"\n💾 Mejores hiperparámetros guardados: {archivo_params}")

# ==================== EVALUACIÓN CON MODELOS OPTIMIZADOS ====================
print("\n" + "="*70)
print("📈 PASO 3: EVALUACIÓN DE MODELOS OPTIMIZADOS")
print("="*70)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

resultados_finales = {}

print("\n🔄 Evaluando modelos optimizados en conjunto de test...")

for nombre, modelo in mejores_modelos.items():
    print(f"\n   📊 {nombre}...")
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    if hasattr(modelo, 'predict_proba'):
        y_proba = modelo.predict_proba(X_test)[:, 1]
    else:
        y_proba = modelo.decision_function(X_test)
    
    # Métricas
    resultados_finales[nombre] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"      ✅ Accuracy: {resultados_finales[nombre]['accuracy']:.4f}")
    print(f"      ✅ Precision: {resultados_finales[nombre]['precision']:.4f}")
    print(f"      ✅ Recall: {resultados_finales[nombre]['recall']:.4f}")
    print(f"      ✅ F1-Score: {resultados_finales[nombre]['f1_score']:.4f}")
    print(f"      ✅ ROC-AUC: {resultados_finales[nombre]['roc_auc']:.4f}")

# ==================== COMPARACIÓN FINAL ====================
print("\n" + "="*70)
print("🏆 PASO 4: COMPARACIÓN Y SELECCIÓN DEL MEJOR MODELO")
print("="*70)

# Crear DataFrame comparativo
df_comparacion = pd.DataFrame(resultados_finales).T
df_comparacion = df_comparacion.sort_values('f1_score', ascending=False)

print("\n📊 RANKING DE MODELOS (ordenado por F1-Score):")
print("="*70)
print(df_comparacion.to_string())

# Identificar mejor modelo
mejor_modelo_nombre = df_comparacion.index[0]
mejor_modelo = mejores_modelos[mejor_modelo_nombre]

print(f"\n🥇 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   F1-Score: {df_comparacion.loc[mejor_modelo_nombre, 'f1_score']:.4f}")
print(f"   Accuracy: {df_comparacion.loc[mejor_modelo_nombre, 'accuracy']:.4f}")
print(f"   ROC-AUC: {df_comparacion.loc[mejor_modelo_nombre, 'roc_auc']:.4f}")

# Guardar comparación
archivo_comp = os.path.join(carpeta_optimizacion, 'comparacion_modelos_optimizados.csv')
df_comparacion.to_csv(archivo_comp)
print(f"\n💾 Comparación guardada: {archivo_comp}")

# Guardar mejor modelo
archivo_mejor = os.path.join(carpeta_optimizacion, 'mejor_modelo_optimizado.pkl')
with open(archivo_mejor, 'wb') as f:
    pickle.dump(mejor_modelo, f)
print(f"💾 Mejor modelo guardado: {archivo_mejor}")

# ==================== VISUALIZACIONES ====================
print("\n📊 Generando visualizaciones comparativas...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
fig.suptitle('Optimización y Comparación de Modelos ML', fontsize=18, fontweight='bold')

# 1. Comparación Cross-Validation
ax1 = fig.add_subplot(gs[0, :2])
modelos_nombres = list(resultados_cv.keys())
cv_f1_means = [resultados_cv[m]['f1_mean'] for m in modelos_nombres]
cv_f1_stds = [resultados_cv[m]['f1_std'] for m in modelos_nombres]

x_pos = np.arange(len(modelos_nombres))
ax1.bar(x_pos, cv_f1_means, yerr=cv_f1_stds, capsize=5, 
       color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(modelos_nombres, rotation=45, ha='right')
ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax1.set_title('Cross-Validation (5-Folds) - F1-Score', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 1])

# Agregar valores
for i, (mean, std) in enumerate(zip(cv_f1_means, cv_f1_stds)):
    ax1.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
            ha='center', fontsize=9, fontweight='bold')

# 2. Mejora con Hyperparameter Tuning
ax2 = fig.add_subplot(gs[0, 2])
mejora = []
for nombre in modelos_nombres:
    cv_score = resultados_cv[nombre]['f1_mean']
    tuned_score = resultados_tuning[nombre]['mejor_score']
    mejora.append((tuned_score - cv_score) * 100)

colors_mejora = ['green' if m > 0 else 'red' for m in mejora]
ax2.barh(range(len(modelos_nombres)), mejora, color=colors_mejora, edgecolor='black')
ax2.set_yticks(range(len(modelos_nombres)))
ax2.set_yticklabels(modelos_nombres)
ax2.set_xlabel('Mejora (%)', fontsize=11, fontweight='bold')
ax2.set_title('Mejora con Tuning', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.3, axis='x')

# Agregar valores
for i, val in enumerate(mejora):
    color = 'green' if val > 0 else 'red'
    ax2.text(val + 0.1, i, f'{val:+.2f}%', va='center', 
            fontweight='bold', color=color, fontsize=9)

# 3. Comparación Final de Métricas
ax3 = fig.add_subplot(gs[1, :])
metricas = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
x = np.arange(len(modelos_nombres))
width = 0.15

for i, metrica in enumerate(metricas):
    valores = [resultados_finales[m][metrica] for m in modelos_nombres]
    ax3.bar(x + i*width, valores, width, label=metrica.replace('_', ' ').title())

ax3.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Comparación Final - Todas las Métricas', fontsize=14, fontweight='bold')
ax3.set_xticks(x + width * 2)
ax3.set_xticklabels(modelos_nombres, rotation=45, ha='right')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim([0, 1.1])

# 4. Tiempos de Entrenamiento
ax4 = fig.add_subplot(gs[2, 0])
tiempos_cv = [resultados_cv[m]['tiempo'] for m in modelos_nombres]
tiempos_tuning = [resultados_tuning[m]['tiempo'] for m in modelos_nombres]

x_pos = np.arange(len(modelos_nombres))
ax4.bar(x_pos - 0.2, tiempos_cv, 0.4, label='CV (5-folds)', color='lightblue')
ax4.bar(x_pos + 0.2, tiempos_tuning, 0.4, label='Hyperparameter Tuning', color='salmon')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(modelos_nombres, rotation=45, ha='right')
ax4.set_ylabel('Tiempo (segundos)', fontsize=11, fontweight='bold')
ax4.set_title('Tiempos de Entrenamiento', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Ranking Final
ax5 = fig.add_subplot(gs[2, 1])
f1_scores = df_comparacion['f1_score'].values
colores_rank = ['gold', 'silver', '#CD7F32', 'lightblue'][:len(modelos_nombres)]

bars = ax5.barh(range(len(df_comparacion)), f1_scores, 
               color=colores_rank, edgecolor='black')
ax5.set_yticks(range(len(df_comparacion)))
ax5.set_yticklabels(df_comparacion.index)
ax5.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax5.set_title('🏆 Ranking Final por F1-Score', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
ax5.set_xlim([0, 1])

# Agregar medallas y valores
medallas = ['🥇', '🥈', '🥉', '4️⃣']
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    medalla = medallas[i] if i < len(medallas) else ''
    ax5.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{medalla} {score:.4f}', va='center', fontweight='bold')

# 6. Tabla Resumen del Mejor Modelo
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

mejor_info = resultados_finales[mejor_modelo_nombre]
mejor_params = resultados_tuning[mejor_modelo_nombre]['mejores_params']

tabla_data = [
    ['Métrica', 'Valor'],
    ['', ''],
    ['🏆 Modelo', mejor_modelo_nombre],
    ['', ''],
    ['Accuracy', f"{mejor_info['accuracy']:.4f}"],
    ['Precision', f"{mejor_info['precision']:.4f}"],
    ['Recall', f"{mejor_info['recall']:.4f}"],
    ['F1-Score', f"{mejor_info['f1_score']:.4f}"],
    ['ROC-AUC', f"{mejor_info['roc_auc']:.4f}"],
]

tabla = ax6.table(cellText=tabla_data, cellLoc='left', loc='center',
                 colWidths=[0.55, 0.45])
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 2.5)

# Estilo
for i in range(len(tabla_data)):
    for j in range(2):
        cell = tabla[(i, j)]
        if i == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        elif i == 2:
            cell.set_facecolor('#FFD700')
            cell.set_text_props(weight='bold')

ax6.set_title('Mejor Modelo - Métricas', fontsize=12, fontweight='bold', pad=20)

# Guardar visualización
archivo_viz = os.path.join(carpeta_optimizacion, 'optimizacion_y_comparacion.png')
plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
print(f"✅ Visualización guardada: {archivo_viz}")
plt.close()

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print("✅ OPTIMIZACIÓN COMPLETADA")
print("="*70)

print(f"\n📁 Archivos generados en: {carpeta_optimizacion}")
print("   📄 resultados_cross_validation.csv")
print("   📄 mejores_hiperparametros.json")
print("   📄 comparacion_modelos_optimizados.csv")
print("   🤖 mejor_modelo_optimizado.pkl")
print("   📊 optimizacion_y_comparacion.png")

print(f"\n🏆 MODELO SELECCIONADO: {mejor_modelo_nombre}")
print(f"   F1-Score: {mejor_info['f1_score']:.4f}")
print(f"   Mejores parámetros: {mejor_params}")

print("\n🎉 ¡Listo para hacer predicciones con el mejor modelo!")