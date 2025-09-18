import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, df):
        self.df = df
    
    def linea_tiempo(self):
        if self.df is None:
            raise ValueError("Llama primero a create_dataset().")
        self.df.sort_values(["CustomerID", "InvoiceDate", "Description"], kind="mergesort", inplace=True)
        return self.df
    
    def create_features(self):
        self.df["Revenue"] = self.df["Quantity"] * self.df["UnitPrice"]
        