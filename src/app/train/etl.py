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
        self.df = None

    def create_dataset(self, include_target: bool = True, target_prefix: str = "target_") -> pd.DataFrame:
        ds = ucimlrepo.fetch_ucirepo(id=352)
        X = ds.data.features.copy()     # Online Retail no trae y
        self.df = X
        return self.df
    

    
        

     

    