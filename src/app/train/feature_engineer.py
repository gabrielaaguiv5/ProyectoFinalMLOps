import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def create_features(self):
        self.df["Revenue"] = self.df["Quantity"] * self.df["UnitPrice"]
        return self.df

    def linea_tiempo(self):
        if self.df is None:
            raise ValueError("Llama primero a create_dataset().")
        self.df.sort_values(["CustomerID", "InvoiceDate", "Description"], kind="mergesort", inplace=True)
        return self.df

    def historial_compra(g: pd.DataFrame) -> pd.DataFrame:
        cp = (
            g.groupby(["Description", "InvoiceDate", "Country"], as_index=False)
             .agg(Quantity=("Quantity", "sum"),
                  Revenue=("Revenue", "sum"),
                  UnitPrice=("UnitPrice", "mean"))
             .sort_values(["InvoiceDate", "Description"], kind="mergesort")
             .reset_index(drop=True)
        )
        cp["CustomerID"] = g.name

        cp["n_past_invoices"] = np.arange(len(cp), dtype=int)
        cp["prev_date"] = cp["InvoiceDate"].shift(1)
        cp["recency_days"] = ((cp["InvoiceDate"] - cp["prev_date"]).dt.days
                              .fillna(9999).astype(int))
        cp["spend_prior"] = cp["Revenue"].cumsum() - cp["Revenue"]
        cp["qty_prior"] = cp["Quantity"].cumsum() - cp["Quantity"]

        denom = cp["n_past_invoices"].replace(0, np.nan)
        cp["avg_ticket_prior"] = (cp["spend_prior"] / denom).fillna(0.0)
        cp["avg_qty_per_invoice_prior"] = (cp["qty_prior"] / denom).fillna(0.0)

        cp["next_date"] = cp["InvoiceDate"].shift(-1)
        cp["days_to_next"] = (cp["next_date"] - cp["InvoiceDate"]).dt.days
        cp["y_repurchase_30d"] = (
            (cp["days_to_next"] <= 30) & (~cp["next_date"].isna())
        ).astype(int)
        return cp

    def historial_compra_cliente(self) -> pd.DataFrame:
        if self.df is None: raise ValueError("Primero llama a set_df(df).")
        out = (
            self.df.groupby("CustomerID", group_keys=False)
                .apply(self._historial_compra_one)
                .reset_index(drop=True)
        )
        self.df = out
        return self.df

    def run(self) -> pd.DataFrame:
        self.create_features()
        self.linea_tiempo()
        self.historial_compra_cliente()
        return self.df