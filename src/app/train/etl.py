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

    def date_tipo(self) -> pd.DataFrame:
        if self.df is None: raise ValueError("Llama primero a create_dataset().")
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors="coerce")
        self.df = self.df.dropna(subset=["InvoiceDate"])
        return self.df

    def limpieza_datos(self) -> pd.DataFrame:
        if self.df is None: raise ValueError("Llama primero a create_dataset().")
        df = self.df
        df = df[(df["UnitPrice"] > 0) & (df["Quantity"] > 0)]
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        df = df.dropna(subset=["CustomerID"]).copy()
        df["CustomerID"] = df["CustomerID"].astype(int)
        self.df = df
        return self.df

    def run_etl(self) -> pd.DataFrame:
        self.invoice_tipo()
        self.date_tipo()
        self.limpieza_datos()
        return self.df
    
        

     

    