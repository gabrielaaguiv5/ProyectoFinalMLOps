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

    def create_dataset(self, include_target: bool = True, target_prefix: str = "target_") -> pd.DataFrame:
        ds = ucimlrepo.fetch_ucirepo(id=352)
        X = ds.data.features.copy()          # <-- ya es DataFrame
        y = ds.data.targets                  # puede ser None o DataFrame/Series

        if include_target and y is not None:
            if not isinstance(y, pd.DataFrame):
                y = pd.DataFrame(y)
            # Asegura nombres de columnas para y
            if any(c is None or c == "" for c in y.columns):
                y.columns = [f"{target_prefix}{i}" for i in range(y.shape[1])]
            # Evita choques de nombres con X
            y = y.rename(columns=lambda c: c if c not in X.columns else f"{c}_target")
            return pd.concat([X, y], axis=1)

        # Para Online Retail (id=352), y es None → devolvemos X
        return X
    
    def Invoice_Tipo(self):
        self["InvoiceNo"] = self["InvoiceNo"].astype(str)

    def Date_Tipo(self):
        self["InvoiceDate"] = pd.to_datetime(self["InvoiceDate"], errors="coerce")
        self = self.dropna(subset=["InvoiceDate"])

    def limpieza_datos_cancelados(self):
        self = self[(self["UnitPrice"] > 0) & (self["Quantity"] > 0)]
        self = self[~self["InvoiceNo"].str.startswith("C")]

    def depuracion_datos_clientes(self):
        self = self.dropna(subset=["CustomerID"]).copy()
        self["CustomerID"] = self["CustomerID"].astype(int)

     

    