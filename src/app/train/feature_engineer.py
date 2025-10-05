import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def _to_datetime(s: pd.Series) -> pd.Series:
        if not np.issubdtype(s.dtype, np.datetime64):
            return pd.to_datetime(s, errors="coerce", utc=False)
        return s

    def create_features(self):
        self.df["InvoiceDate"] = self._to_datetime(self.df["InvoiceDate"])
        self.df["Revenue"] = self.df["Quantity"] * self.df["UnitPrice"]
        return self.df

    def _collapse_to_visit_level(self, g: pd.DataFrame) -> pd.DataFrame:
        """
        Colapsa las líneas del cliente a nivel 'visita' (idealmente InvoiceNo).
        - Si existe InvoiceNo: usa InvoiceNo como id de visita.
        - Si no existe: agrupa por fecha exacta de la factura (redondeada a minuto) para aproximar una visita.
        """
        if "InvoiceNo" in g.columns:
            # Mantiene la marca temporal de la visita usando el mínimo InvoiceDate de ese InvoiceNo
            visit = (
                g.groupby("InvoiceNo", as_index=False)
                 .agg(
                     InvoiceDate=("InvoiceDate", "min"),
                     Quantity=("Quantity", "sum"),
                     Revenue=("Revenue", "sum"),
                     UnitPrice=("UnitPrice", "mean"),
                     Country=("Country", "first"),
                 )
                 .sort_values("InvoiceDate", kind="mergesort")
                 .reset_index(drop=True)
            )
        else:
            # Fallback: agrupar por (InvoiceDate) exacto -> “visita” aproximada
            # Si tienes varias compras exactamente en el mismo segundo, esto sigue siendo razonable.
            visit = (
                g.groupby("InvoiceDate", as_index=False)
                 .agg(
                     Quantity=("Quantity", "sum"),
                     Revenue=("Revenue", "sum"),
                     UnitPrice=("UnitPrice", "mean"),
                     Country=("Country", "first"),
                 )
                 .sort_values("InvoiceDate", kind="mergesort")
                 .reset_index(drop=True)
            )

        visit["CustomerID"] = g.name
        return visit

    def _build_customer_history(self, g: pd.DataFrame) -> pd.DataFrame:
        """
        Construye features a nivel visita y etiqueta y_repurchase_30d mirando hacia adelante.
        """
        # 1) Colapsar a nivel visita
        cp = self._collapse_to_visit_level(g)

        # 2) Índice de visitas previas (0,1,2,...)
        cp["n_past_invoices"] = np.arange(len(cp), dtype=int)

        # 3) Recency y acumulados
        cp["prev_date"] = cp["InvoiceDate"].shift(1)
        cp["recency_days"] = (cp["InvoiceDate"] - cp["prev_date"]).dt.days
        cp["recency_days"] = cp["recency_days"].fillna(9999).astype("float64")  # float64 p/ firma MLflow

        cp["spend_prior"] = cp["Revenue"].cumsum() - cp["Revenue"]
        cp["qty_prior"] = cp["Quantity"].cumsum() - cp["Quantity"]

        denom = cp["n_past_invoices"].replace(0, np.nan)
        cp["avg_ticket_prior"] = (cp["spend_prior"] / denom).fillna(0.0)
        cp["avg_qty_per_invoice_prior"] = (cp["qty_prior"] / denom).fillna(0.0)

        # 4) Próxima visita y target hacia adelante
        cp["next_date"] = cp["InvoiceDate"].shift(-1)
        cp["days_to_next"] = (cp["next_date"] - cp["InvoiceDate"]).dt.days
        cp["y_repurchase_30d"] = ((cp["days_to_next"] <= 30) & (~cp["next_date"].isna())).astype(int)

        return cp

    # -----------------------------
    # API pública
    # -----------------------------
    def create_features(self):
        # Tipar fecha y calcular Revenue
        self.df["InvoiceDate"] = self._to_datetime(self.df["InvoiceDate"])
        self.df["Revenue"] = self.df["Quantity"] * self.df["UnitPrice"]
        return self.df

    def linea_tiempo(self):
        if self.df is None:
            raise ValueError("Llama primero a create_features().")
        # Para ordenar estable, a nivel cliente-visita (InvoiceNo si existe)
        sort_cols = ["CustomerID", "InvoiceDate"]
        if "InvoiceNo" in self.df.columns:
            sort_cols = ["CustomerID", "InvoiceDate", "InvoiceNo"]
        self.df.sort_values(sort_cols, kind="mergesort", inplace=True)
        return self.df

    def historial_compra_cliente(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Primero llama a create_features().")
        out = (
            self.df.groupby("CustomerID", group_keys=False)
                   .apply(self._build_customer_history, include_groups=False)
                   .reset_index(drop=True)
        )
        self.df = out
        return self.df

    def aplicar_censura_derecha(self) -> pd.DataFrame:
        """
        Elimina visitas de los últimos 30 días del rango temporal global,
        porque no puedes observar una recompra posterior.
        """
        if self.df is None:
            raise ValueError("Primero corre historial_compra_cliente()")
        max_date = self.df["InvoiceDate"].max()
        cutoff = max_date - pd.Timedelta(days=30)
        self.df = self.df[self.df["InvoiceDate"] <= cutoff].reset_index(drop=True)
        return self.df

    def cast_numeric_float64(self) -> pd.DataFrame:
        """
        Asegura float64 en numéricas para evitar problemas de schema en MLflow
        cuando haya valores faltantes en inferencia.
        """
        num_cols = [
            "recency_days","n_past_invoices","spend_prior","qty_prior",
            "avg_ticket_prior","avg_qty_per_invoice_prior","UnitPrice","Quantity","Revenue",
            "days_to_next"
        ]
        present = [c for c in num_cols if c in self.df.columns]
        self.df[present] = self.df[present].astype("float64")
        return self.df

    def run(self) -> pd.DataFrame:
        self.create_features()
        self.linea_tiempo()
        self.historial_compra_cliente()
        self.aplicar_censura_derecha()
        self.cast_numeric_float64()
        return self.df

    def save_dataset(self, path: str):
        if self.df is None:
            raise ValueError("Primero corre .run() para procesar el dataset.")
        self.df.to_parquet(path, index=False)
        print(f"Dataset guardado en {path}")