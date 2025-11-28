import pickle
import numpy as np
import os

# Cargar modelo
carpeta_modelo = 'Data/Procesados/MLFF'

with open(os.path.join(carpeta_modelo, 'modelo.pkl'), 'rb') as f:
    modelo = pickle.load(f)
with open(os.path.join(carpeta_modelo, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(carpeta_modelo, 'train_stats.pkl'), 'rb') as f:
    stats = pickle.load(f)
with open(os.path.join(carpeta_modelo, 'features.txt'), 'r') as f:
    features = f.read().splitlines()

def predecir_amenaza(sismos_total, magnitud_max, magnitud_media, magnitud_std):
    """
    Predice la amenaza sísmica de un municipio.
    
    Returns:
        tuple: (amenaza, probabilidades, confianza)
        - amenaza: str ('BAJA', 'MEDIA', 'ALTA')
        - probabilidades: array([prob_baja, prob_media, prob_alta])
        - confianza: float (0-1)
    """
    # Crear features
    X_dict = {
        'sismos_total': sismos_total,
        'magnitud_media': magnitud_media,
        'magnitud_std': magnitud_std,
        'magnitud_mediana': magnitud_media,
        'densidad': sismos_total / stats['max_sismos'],
        'actividad_alta': 1 if sismos_total > stats['q75_sismos'] else 0,
        'variabilidad_alta': 1 if magnitud_std > stats['median_std'] else 0
    }
    
    X = np.array([[X_dict[f] for f in features]])
    X_scaled = scaler.transform(X)
    
    pred = modelo.predict(X_scaled)[0]
    probs = modelo.predict_proba(X_scaled)[0]
    
    clases = ['BAJA', 'MEDIA', 'ALTA']
    return clases[pred], probs, max(probs)


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo 1
    amenaza, probs, conf = predecir_amenaza(
        sismos_total=45,
        magnitud_max=2.8,
        magnitud_media=2.1,
        magnitud_std=0.5
    )
    
    print(f"Amenaza: {amenaza}")
    print(f"Probabilidades: BAJA={probs[0]:.3f}, MEDIA={probs[1]:.3f}, ALTA={probs[2]:.3f}")
    print(f"Confianza: {conf:.3f}")