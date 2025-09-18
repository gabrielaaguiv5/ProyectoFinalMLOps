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
    
        print(df.metadata) 
        
        print(df.variables) 
        return df, df.metadata, df.variables
    
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

     

    