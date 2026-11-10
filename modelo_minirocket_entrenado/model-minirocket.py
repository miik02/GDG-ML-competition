from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
from competition_api import ABSTAIN, Sample

# Evitar warnings de scikit-learn
warnings.filterwarnings('ignore')

HERE = Path(__file__).parent 

# Umbrales del ABSTAIN (Puedes probar a ponerlos todos a 0.0 para probar la versión "SIN ABSTAIN")
UMBRALES = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,
    4: 0.0, 5: 0.0, 6: 0.6, 7: 0.0
}

def imputar_nans(serie):
    df = pd.DataFrame(serie.T)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    return df.values.T

class Model:
    def __init__(self):
        # La fase de setup permite cargar los archivos. Tiene límite de 120s.
        ruta_pesos = HERE / "weights.pkl"
        pesos = joblib.load(ruta_pesos)
        self.minirocket = pesos["minirocket"]
        self.clasificador = pesos["ridge"]

    def predict(self, x: Sample) -> int:
        # 1. Extraemos y limpiamos la señal de (4, 256)
        raw_serie = x.series
        serie_limpia = imputar_nans(raw_serie)
        
        # 2. La metemos en una dimensión extra porque MiniROCKET espera (1, C, L) y en float64
        serie_lista = serie_limpia.astype(np.float64)[np.newaxis, ...]
        
        # 3. Transformación (10k características)
        trans_serie = self.minirocket.transform(serie_lista)
        
        # 4. Predicción con probabilidades
        probabilidades = self.clasificador.predict_proba(trans_serie)[0]
        clase_predicha = int(np.argmax(probabilidades))
        confianza = probabilidades[clase_predicha]
        
        # 5. Abstain (Si no supera el umbral, pasamos)
        if confianza >= UMBRALES[clase_predicha]:
            return clase_predicha
        else:
            return ABSTAIN