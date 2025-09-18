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
from ucimlrepo import fetch_ucirepo, list_available_datasets
import numpy as np
import pandas as pd


class UserGenerator:
    
    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        self.seed = seed

    def create_dataset(self):
        df = fetch_ucirepo(id=352) 
        
        X = df.data.features 
        y = df.data.targets 
    
        print(df.metadata) 
        
        print(df.variables) 
        return X, y, df.metadata, df.variables
        