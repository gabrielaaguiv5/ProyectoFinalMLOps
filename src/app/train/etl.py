### EDA
### Mantener la carpeta de data con CSV sin modificar
### Si denttro de la carpte data exite retail, no me descarga nadagit
###
#!/usr/bin/env python3
"""
Script para generar datos sintéticos de usuarios para targeting de promociones.
Este script simula un dataset realista para el problema de identificar qué usuarios
deben recibir promociones basado en su comportamiento transaccional y perfil.
"""

import json
import random
from datetime import datetime, timedelta
import ucimlrepo
import numpy as np
import pandas as pd
from types import SimpleNamespace



class UserGenerator:
    
    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        self.seed = seed

    def create_dataset(self):
        ds = ucimlrepo.fetch_ucirepo(id=352)

        # Si por algún wrapper llega tupla: (X, y, metadata, variables), la envolvemos.
        if isinstance(ds, tuple):
            X, y, metadata, variables = ds
            ds = SimpleNamespace(
                data=SimpleNamespace(features=X, targets=y),
                metadata=metadata,
                variables=variables,
            )
        return ds
    