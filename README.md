# 🌎 Predicción de Amenaza Sísmica en Colombia

Modelo de machine learning para clasificar el nivel de amenaza sísmica (BAJA, MEDIA, ALTA) por municipio, entrenado con datos históricos del Servicio Geológico Colombiano.

---

## 📌 Descripción del problema

Colombia es uno de los países con mayor actividad sísmica en América Latina. Este proyecto busca responder una pregunta concreta: **dado el historial sísmico de un municipio, ¿cuál es su nivel de amenaza?**

El modelo toma estadísticas agregadas de sismos registrados en un municipio y predice si la amenaza es **BAJA**, **MEDIA** o **ALTA**, junto con las probabilidades asociadas a cada clase.

---

## 📂 Fuente de datos

- **Servicio Geológico Colombiano (SGC)** — base de datos histórica de sismos en Colombia
- Los datos fueron sometidos a un proceso de validación y limpieza: revisión de duplicados, valores nulos y consistencia de registros
- La base se encontró en buenas condiciones, sin errores significativos, lo que refleja los estándares de calidad del SGC

---

## ⚙️ Features utilizadas

| Feature | Descripción |
|---|---|
| `sismos_total` | Número total de sismos registrados en el municipio |
| `magnitud_media` | Magnitud promedio de los sismos |
| `magnitud_std` | Desviación estándar de la magnitud |
| `magnitud_mediana` | Mediana de la magnitud |
| `densidad` | Sismos totales normalizados por el máximo histórico |
| `actividad_alta` | 1 si el municipio supera el percentil 75 de actividad |
| `variabilidad_alta` | 1 si la variabilidad de magnitud supera la mediana |

---

## 🤖 Modelos evaluados

Se entrenaron y compararon múltiples modelos de clasificación:

- **Random Forest** ✅ *(mejor desempeño)*
- Árbol de Decisión
- K-Nearest Neighbors (KNN)
- Otros clasificadores del ecosistema scikit-learn

La selección final se basó en métricas de evaluación sobre conjunto de prueba, incluyendo **matriz de confusión**, accuracy y análisis por clase.

---

## ⚠️ Limitación identificada: sesgo por clases desbalanceadas

En Colombia, la gran mayoría de sismos históricos son de magnitud baja. Esto genera un **desbalance de clases** en los datos de entrenamiento: el modelo tiene poca exposición a eventos de alta magnitud, lo que reduce su capacidad predictiva para la clase ALTA.

Este es un problema conocido en datasets sísmicos de zonas de actividad moderada y representa una oportunidad de mejora futura mediante técnicas como SMOTE o pesos de clase ajustados.

---

## 🛠️ Tecnologías utilizadas

- Python 3
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn
- pickle (serialización del modelo)

---

## 🚀 Cómo usar el modelo

```python
from predictor import predecir_amenaza

amenaza, probs, confianza = predecir_amenaza(
    sismos_total=45,
    magnitud_max=2.8,
    magnitud_media=2.1,
    magnitud_std=0.5
)

print(f"Amenaza: {amenaza}")
print(f"Probabilidades: BAJA={probs[0]:.3f}, MEDIA={probs[1]:.3f}, ALTA={probs[2]:.3f}")
print(f"Confianza: {confianza:.3f}")
```

---

## 👥 Autores

- Samir Centeno — [@scentenoo](https://github.com/scentenoo)
- [Nombre de tu compañero] — Universidad Nacional de Colombia, Sede Medellín

---

## 📄 Contexto académico

Proyecto final desarrollado en la Universidad Nacional de Colombia (Sede Medellín) como entrega de curso. Los datos provienen exclusivamente de fuentes públicas del Servicio Geológico Colombiano.
