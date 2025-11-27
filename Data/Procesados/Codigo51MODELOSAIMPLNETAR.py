import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🤖 FASE 5: MODELADO DE MACHINE LEARNING")
print("="*70)

# ==================== CONFIGURACIÓN ====================
carpeta_ml = 'Data/Procesados/ML'
carpeta_modelos = 'Data/Procesados/ML/Modelos'
os.makedirs(carpeta_modelos, exist_ok=True)

# ==================== CARGAR DATOS ====================
print("\n📥 Cargando datos preprocesados...")

try:
    X_train = np.load(os.path.join(carpeta_ml, 'X_train.npy'))
    y_train = np.load(os.path.join(carpeta_ml, 'y_train.npy'))
    X_test = np.load(os.path.join(carpeta_ml, 'X_test.npy'))
    y_test = np.load(os.path.join(carpeta_ml, 'y_test.npy'))
    
    print(f"✅ Datos cargados exitosamente")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_test: {y_test.shape}")
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    print("   Asegúrate de haber ejecutado preparacion_ml.py primero")
    exit()

# ==================== 5.1 DEFINIR MODELOS ====================
print("\n" + "="*70)
print("🤖 5.1 MODELOS A IMPLEMENTAR")
print("="*70)

modelos = {
    '1. K-NN (Baseline)': KNeighborsClassifier(n_neighbors=5),
    '2. Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    '3. XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    '4. SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

print("\nModelos configurados:")
for nombre in modelos.keys():
    print(f"   ✅ {nombre}")

# ==================== ENTRENAR Y EVALUAR MODELOS ====================
print("\n" + "="*70)
print("🏋️ ENTRENANDO MODELOS...")
print("="*70)

resultados = {}
modelos_entrenados = {}

for nombre, modelo in modelos.items():
    print(f"\n🔄 Entrenando: {nombre}")
    
    # Entrenar
    modelo.fit(X_train, y_train)
    print(f"   ✅ Entrenamiento completado")
    
    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Probabilidades (para ROC)
    if hasattr(modelo, 'predict_proba'):
        y_proba_test = modelo.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = modelo.decision_function(X_test)
    
    # Calcular métricas
    metricas = {
        'accuracy_train': accuracy_score(y_train, y_pred_train),
        'accuracy_test': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1_score': f1_score(y_test, y_pred_test),
        'roc_auc': roc_auc_score(y_test, y_proba_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'y_pred_test': y_pred_test,
        'y_proba_test': y_proba_test
    }
    
    resultados[nombre] = metricas
    modelos_entrenados[nombre] = modelo
    
    print(f"   📊 Accuracy Test: {metricas['accuracy_test']:.4f}")
    print(f"   📊 F1-Score: {metricas['f1_score']:.4f}")
    print(f"   📊 ROC-AUC: {metricas['roc_auc']:.4f}")

# ==================== GUARDAR MODELOS ====================
print("\n💾 Guardando modelos entrenados...")

for nombre, modelo in modelos_entrenados.items():
    nombre_archivo = nombre.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
    archivo_modelo = os.path.join(carpeta_modelos, f'modelo_{nombre_archivo}.pkl')
    with open(archivo_modelo, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"   ✅ {nombre} guardado")

# ==================== COMPARACIÓN DE MODELOS ====================
print("\n" + "="*70)
print("📊 COMPARACIÓN DE MODELOS")
print("="*70)

# Crear DataFrame con resultados
df_comparacion = pd.DataFrame({
    'Modelo': list(resultados.keys()),
    'Accuracy (Train)': [r['accuracy_train'] for r in resultados.values()],
    'Accuracy (Test)': [r['accuracy_test'] for r in resultados.values()],
    'Precision': [r['precision'] for r in resultados.values()],
    'Recall': [r['recall'] for r in resultados.values()],
    'F1-Score': [r['f1_score'] for r in resultados.values()],
    'ROC-AUC': [r['roc_auc'] for r in resultados.values()]
})

print("\n" + df_comparacion.to_string(index=False))

# Guardar comparación
archivo_comparacion = os.path.join(carpeta_modelos, 'comparacion_modelos.csv')
df_comparacion.to_csv(archivo_comparacion, index=False)
print(f"\n💾 Comparación guardada: {archivo_comparacion}")

# Identificar mejor modelo
mejor_modelo_idx = df_comparacion['F1-Score'].idxmax()
mejor_modelo = df_comparacion.loc[mejor_modelo_idx, 'Modelo']
print(f"\n🏆 MEJOR MODELO: {mejor_modelo}")
print(f"   F1-Score: {df_comparacion.loc[mejor_modelo_idx, 'F1-Score']:.4f}")

# ==================== VISUALIZACIONES ====================
print("\n📊 Generando visualizaciones...")

# Crear figura grande
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
fig.suptitle('Evaluación de Modelos de Machine Learning', fontsize=20, fontweight='bold')

# ==================== 1. COMPARACIÓN DE MÉTRICAS ====================
ax1 = fig.add_subplot(gs[0, :2])

metricas_comp = ['Accuracy (Test)', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(modelos))
width = 0.15

for i, metrica in enumerate(metricas_comp):
    valores = df_comparacion[metrica].values
    ax1.bar(x + i*width, valores, width, label=metrica)

ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels([m.split('.')[1].strip() for m in modelos.keys()], rotation=45, ha='right')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 1.1])

# ==================== 2. ACCURACY TRAIN VS TEST ====================
ax2 = fig.add_subplot(gs[0, 2:])

x_pos = np.arange(len(modelos))
acc_train = df_comparacion['Accuracy (Train)'].values
acc_test = df_comparacion['Accuracy (Test)'].values

ax2.bar(x_pos - 0.2, acc_train, 0.4, label='Train', color='steelblue')
ax2.bar(x_pos + 0.2, acc_test, 0.4, label='Test', color='coral')

ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy: Train vs Test (Detección de Overfitting)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([m.split('.')[1].strip() for m in modelos.keys()], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1.1])

# Agregar valores
for i, (train, test) in enumerate(zip(acc_train, acc_test)):
    diff = train - test
    color = 'red' if diff > 0.1 else 'green'
    ax2.text(i, max(train, test) + 0.02, f'Δ={diff:.3f}', 
            ha='center', fontsize=8, color=color, fontweight='bold')

# ==================== 3-6. MATRICES DE CONFUSIÓN ====================
cm_axes = [
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[1, 3])
]

for idx, (nombre, metricas) in enumerate(resultados.items()):
    cm = metricas['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Baja/Media', 'Alta'],
               yticklabels=['Baja/Media', 'Alta'],
               ax=cm_axes[idx], cbar=True)
    
    nombre_corto = nombre.split('.')[1].strip()
    cm_axes[idx].set_title(f'Matriz de Confusión\n{nombre_corto}', fontsize=12, fontweight='bold')
    cm_axes[idx].set_ylabel('Real', fontsize=10)
    cm_axes[idx].set_xlabel('Predicción', fontsize=10)

# ==================== 7. CURVAS ROC ====================
ax_roc = fig.add_subplot(gs[2, :2])

for nombre, metricas in resultados.items():
    fpr, tpr, _ = roc_curve(y_test, metricas['y_proba_test'])
    roc_auc = metricas['roc_auc']
    nombre_corto = nombre.split('.')[1].strip()
    ax_roc.plot(fpr, tpr, linewidth=2, label=f'{nombre_corto} (AUC = {roc_auc:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax_roc.set_title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
ax_roc.legend(loc='lower right')
ax_roc.grid(True, alpha=0.3)

# ==================== 8. F1-SCORE POR MODELO ====================
ax_f1 = fig.add_subplot(gs[2, 2:])

f1_scores = df_comparacion['F1-Score'].values
colores = ['gold' if i == mejor_modelo_idx else 'lightblue' for i in range(len(modelos))]

bars = ax_f1.barh(range(len(modelos)), f1_scores, color=colores, edgecolor='black')
ax_f1.set_yticks(range(len(modelos)))
ax_f1.set_yticklabels([m.split('.')[1].strip() for m in modelos.keys()])
ax_f1.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax_f1.set_title('F1-Score por Modelo (⭐ = Mejor)', fontsize=14, fontweight='bold')
ax_f1.grid(True, alpha=0.3, axis='x')
ax_f1.set_xlim([0, 1])

# Agregar valores y estrella
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    text = f'{score:.4f}'
    if i == mejor_modelo_idx:
        text += ' ⭐'
    ax_f1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
              text, va='center', fontweight='bold')

# ==================== 9. TABLA DE CLASIFICACIÓN (MEJOR MODELO) ====================
ax_tabla = fig.add_subplot(gs[3, :2])
ax_tabla.axis('off')

mejor_metricas = resultados[mejor_modelo]
report = classification_report(y_test, mejor_metricas['y_pred_test'], 
                               target_names=['Baja/Media', 'Alta'],
                               output_dict=True)

tabla_data = [
    ['Clase', 'Precision', 'Recall', 'F1-Score', 'Support'],
    ['', '', '', '', ''],
]

for clase in ['Baja/Media', 'Alta']:
    tabla_data.append([
        clase,
        f"{report[clase]['precision']:.3f}",
        f"{report[clase]['recall']:.3f}",
        f"{report[clase]['f1-score']:.3f}",
        f"{int(report[clase]['support'])}"
    ])

tabla_data.extend([
    ['', '', '', '', ''],
    ['Accuracy', '', '', f"{report['accuracy']:.3f}", ''],
    ['Macro avg', f"{report['macro avg']['precision']:.3f}",
     f"{report['macro avg']['recall']:.3f}",
     f"{report['macro avg']['f1-score']:.3f}", ''],
])

tabla = ax_tabla.table(cellText=tabla_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 2.5)

# Estilo de la tabla
for i in range(len(tabla_data)):
    for j in range(5):
        cell = tabla[(i, j)]
        if i == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        elif i == 1 or i == len(tabla_data) - 3:
            cell.set_facecolor('#E7E6E6')

ax_tabla.set_title(f'Reporte de Clasificación - {mejor_modelo}', 
                  fontsize=12, fontweight='bold', pad=20)

# ==================== 10. RESUMEN GENERAL ====================
ax_resumen = fig.add_subplot(gs[3, 2:])
ax_resumen.axis('off')

resumen_data = [
    ['Métrica', 'Valor'],
    ['', ''],
    ['Mejor Modelo', mejor_modelo.split('.')[1].strip()],
    ['', ''],
    ['Accuracy Test', f"{df_comparacion.loc[mejor_modelo_idx, 'Accuracy (Test)']:.4f}"],
    ['Precision', f"{df_comparacion.loc[mejor_modelo_idx, 'Precision']:.4f}"],
    ['Recall', f"{df_comparacion.loc[mejor_modelo_idx, 'Recall']:.4f}"],
    ['F1-Score', f"{df_comparacion.loc[mejor_modelo_idx, 'F1-Score']:.4f}"],
    ['ROC-AUC', f"{df_comparacion.loc[mejor_modelo_idx, 'ROC-AUC']:.4f}"],
    ['', ''],
    ['Registros Train', f"{len(X_train)}"],
    ['Registros Test', f"{len(X_test)}"],
    ['Features', f"{X_train.shape[1]}"],
]

tabla_resumen = ax_resumen.table(cellText=resumen_data, cellLoc='left', loc='center',
                                colWidths=[0.6, 0.4])
tabla_resumen.auto_set_font_size(False)
tabla_resumen.set_fontsize(11)
tabla_resumen.scale(1, 2.5)

# Estilo
for i in range(len(resumen_data)):
    for j in range(2):
        cell = tabla_resumen[(i, j)]
        if i == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        elif i == 2:
            cell.set_facecolor('#FFD966')
            if j == 1:
                cell.set_text_props(weight='bold')

ax_resumen.set_title('Resumen General', fontsize=12, fontweight='bold', pad=20)

# Guardar visualización
archivo_viz = os.path.join(carpeta_modelos, 'evaluacion_modelos.png')
plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
print(f"✅ Visualización guardada: {archivo_viz}")
plt.close()

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print("✅ MODELADO DE ML COMPLETADO")
print("="*70)

print(f"\n📁 Archivos guardados en: {carpeta_modelos}")
print("   🤖 Modelos entrenados (.pkl)")
print("   📄 comparacion_modelos.csv")
print("   📊 evaluacion_modelos.png")

print(f"\n🏆 MEJOR MODELO: {mejor_modelo}")
print(f"   Accuracy: {df_comparacion.loc[mejor_modelo_idx, 'Accuracy (Test)']:.4f}")
print(f"   Precision: {df_comparacion.loc[mejor_modelo_idx, 'Precision']:.4f}")
print(f"   Recall: {df_comparacion.loc[mejor_modelo_idx, 'Recall']:.4f}")
print(f"   F1-Score: {df_comparacion.loc[mejor_modelo_idx, 'F1-Score']:.4f}")
print(f"   ROC-AUC: {df_comparacion.loc[mejor_modelo_idx, 'ROC-AUC']:.4f}")

print("\n🎉 ¡Modelos listos para hacer predicciones!")