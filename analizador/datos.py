"""Descarga y normalización de datos.

Fuente principal: Financial Modeling Prep (API "stable").
Respaldo: Yahoo Finance (librería yfinance), que además se usa para
insiders y accionistas porque esos datos son de pago en FMP.

Todos los estados financieros se convierten a un formato común:
un DataFrame anual, ordenado del año más antiguo al más reciente,
con estas columnas (en millones no: en unidades de la divisa):

    revenue, operating_income, ebit, ebitda, net_income, eps_diluted,
    shares_diluted, interest_expense, rnd, income_tax, pretax_income,
    total_assets, total_liabilities, total_debt, cash, equity,
    current_assets, current_liabilities, inventory, non_current_assets,
    cfo, capex, fcf, change_wc
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests

FMP_BASE = "https://financialmodelingprep.com/stable"

COLUMNAS = [
    "revenue", "operating_income", "ebit", "ebitda", "net_income", "eps_diluted",
    "shares_diluted", "interest_expense", "rnd", "income_tax", "pretax_income",
    "total_assets", "total_liabilities", "total_debt", "cash", "equity",
    "current_assets", "current_liabilities", "inventory", "non_current_assets",
    "cfo", "capex", "fcf", "change_wc",
]


@dataclass
class DatosEmpresa:
    ticker: str
    fuente: str = ""
    perfil: dict = field(default_factory=dict)
    anual: pd.DataFrame = field(default_factory=pd.DataFrame)
    precios: pd.Series = field(default_factory=pd.Series)  # cierre diario, índice fecha
    directivos: pd.DataFrame = field(default_factory=pd.DataFrame)
    insiders: pd.DataFrame = field(default_factory=pd.DataFrame)
    insiders_resumen: pd.DataFrame = field(default_factory=pd.DataFrame)
    accionistas: dict = field(default_factory=dict)  # % insiders, % instituciones
    institucionales: pd.DataFrame = field(default_factory=pd.DataFrame)
    avisos: list = field(default_factory=list)

    @property
    def nombre(self) -> str:
        return self.perfil.get("nombre") or self.ticker

    @property
    def precio(self) -> float | None:
        p = self.perfil.get("precio")
        if p:
            return float(p)
        if not self.precios.empty:
            return float(self.precios.iloc[-1])
        return None

    @property
    def market_cap(self) -> float | None:
        mc = self.perfil.get("market_cap")
        if mc:
            return float(mc)
        if self.precio and not self.anual.empty:
            acc = self.anual["shares_diluted"].dropna()
            if not acc.empty:
                return self.precio * float(acc.iloc[-1])
        return None


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _primero(d: dict, *claves, defecto=np.nan):
    """Devuelve el primer valor no vacío entre varias claves posibles."""
    for c in claves:
        v = d.get(c)
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
    return defecto


class ErrorFMP(Exception):
    pass


def _fmp(ruta: str, clave: str, **params) -> Any:
    params["apikey"] = clave
    r = requests.get(f"{FMP_BASE}/{ruta}", params=params, timeout=20)
    if r.status_code in (401, 403):
        raise ErrorFMP(
            "FMP ha rechazado la petición (clave no válida o dato no incluido en tu plan)."
        )
    if r.status_code == 429:
        raise ErrorFMP("Has llegado al límite diario de peticiones de FMP.")
    r.raise_for_status()
    datos = r.json()
    if isinstance(datos, dict) and ("Error Message" in datos or "error" in datos):
        raise ErrorFMP(str(datos.get("Error Message") or datos.get("error")))
    return datos


def _fmp_estado(ruta: str, clave: str, ticker: str) -> list:
    """Pide 10 años; si el plan no lo permite, reintenta con 5."""
    for limite in (10, 5):
        try:
            datos = _fmp(ruta, clave, symbol=ticker, period="annual", limit=limite)
            if isinstance(datos, list):
                return datos
        except (ErrorFMP, requests.HTTPError):
            continue
    return []


# ---------------------------------------------------------------------------
# FMP
# ---------------------------------------------------------------------------

def _fmp_anual(ticker: str, clave: str) -> pd.DataFrame:
    inc = _fmp_estado("income-statement", clave, ticker)
    bal = _fmp_estado("balance-sheet-statement", clave, ticker)
    cf = _fmp_estado("cash-flow-statement", clave, ticker)
    if not inc or not bal:
        return pd.DataFrame()

    filas: dict[str, dict] = {}

    for d in inc:
        f = filas.setdefault(d["date"], {})
        f.update(
            revenue=_primero(d, "revenue"),
            operating_income=_primero(d, "operatingIncome"),
            ebit=_primero(d, "ebit", "operatingIncome"),
            ebitda=_primero(d, "ebitda"),
            net_income=_primero(d, "netIncome"),
            eps_diluted=_primero(d, "epsDiluted", "epsdiluted", "eps"),
            shares_diluted=_primero(d, "weightedAverageShsOutDil", "weightedAverageShsOut"),
            interest_expense=_primero(d, "interestExpense"),
            rnd=_primero(d, "researchAndDevelopmentExpenses"),
            income_tax=_primero(d, "incomeTaxExpense"),
            pretax_income=_primero(d, "incomeBeforeTax"),
        )
    for d in bal:
        f = filas.setdefault(d["date"], {})
        f.update(
            total_assets=_primero(d, "totalAssets"),
            total_liabilities=_primero(d, "totalLiabilities"),
            total_debt=_primero(d, "totalDebt"),
            cash=_primero(d, "cashAndCashEquivalents", "cashAndShortTermInvestments"),
            equity=_primero(d, "totalStockholdersEquity", "totalEquity"),
            current_assets=_primero(d, "totalCurrentAssets"),
            current_liabilities=_primero(d, "totalCurrentLiabilities"),
            inventory=_primero(d, "inventory", defecto=0.0),
            non_current_assets=_primero(d, "totalNonCurrentAssets"),
        )
    for d in cf:
        f = filas.setdefault(d["date"], {})
        cfo = _primero(d, "operatingCashFlow", "netCashProvidedByOperatingActivities")
        capex = _primero(d, "capitalExpenditure")
        f.update(
            cfo=cfo,
            capex=capex,
            fcf=_primero(d, "freeCashFlow", defecto=cfo + capex if pd.notna(cfo) and pd.notna(capex) else np.nan),
            change_wc=_primero(d, "changeInWorkingCapital"),
        )

    df = pd.DataFrame.from_dict(filas, orient="index")
    df.index = pd.to_datetime(df.index)
    return _limpiar(df)


def _fmp_perfil(ticker: str, clave: str) -> dict:
    datos = _fmp("profile", clave, symbol=ticker)
    if not datos:
        return {}
    d = datos[0]
    return {
        "nombre": d.get("companyName"),
        "precio": _primero(d, "price", defecto=None),
        "market_cap": _primero(d, "marketCap", "mktCap", defecto=None),
        "beta": _primero(d, "beta", defecto=None),
        "sector": d.get("sector"),
        "industria": d.get("industry"),
        "descripcion": d.get("description"),
        "ceo": d.get("ceo"),
        "pais": d.get("country"),
        "divisa": d.get("currency"),
        "bolsa": d.get("exchange") or d.get("exchangeShortName"),
        "empleados": d.get("fullTimeEmployees"),
        "web": d.get("website"),
        "ipo": d.get("ipoDate"),
    }


def _fmp_precios(ticker: str, clave: str, anios: int = 11) -> pd.Series:
    desde = (pd.Timestamp.today() - pd.DateOffset(years=anios)).strftime("%Y-%m-%d")
    datos = _fmp("historical-price-eod/light", clave, symbol=ticker, **{"from": desde})
    if isinstance(datos, dict):  # formato antiguo
        datos = datos.get("historical", [])
    if not datos:
        return pd.Series(dtype=float)
    df = pd.DataFrame(datos)
    col = "close" if "close" in df else "price"
    s = pd.Series(df[col].astype(float).values, index=pd.to_datetime(df["date"]))
    return s.sort_index()


def _fmp_directivos(ticker: str, clave: str) -> pd.DataFrame:
    datos = _fmp("key-executives", clave, symbol=ticker)
    if not datos:
        return pd.DataFrame()
    df = pd.DataFrame(datos)
    cols = {"name": "Nombre", "title": "Cargo", "pay": "Retribución", "currencyPay": "Divisa",
            "yearBorn": "Año nac.", "titleSince": "En el cargo desde"}
    df = df[[c for c in cols if c in df]].rename(columns=cols)
    return df


# ---------------------------------------------------------------------------
# Yahoo Finance (respaldo + insiders)
# ---------------------------------------------------------------------------

_YF_MAPA = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "eps_diluted": ["Diluted EPS"],
    "shares_diluted": ["Diluted Average Shares", "Basic Average Shares"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "rnd": ["Research And Development"],
    "income_tax": ["Tax Provision"],
    "pretax_income": ["Pretax Income"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "inventory": ["Inventory"],
    "non_current_assets": ["Total Non Current Assets"],
    "cfo": ["Operating Cash Flow"],
    "capex": ["Capital Expenditure"],
    "fcf": ["Free Cash Flow"],
    "change_wc": ["Change In Working Capital"],
}


def _yf_anual(tk) -> pd.DataFrame:
    tablas = []
    for nombre in ("income_stmt", "balance_sheet", "cashflow"):
        try:
            t = getattr(tk, nombre)
            if t is not None and not t.empty:
                tablas.append(t)
        except Exception:
            pass
    if not tablas:
        return pd.DataFrame()
    todo = pd.concat(tablas)
    todo = todo[~todo.index.duplicated()]
    df = pd.DataFrame(index=pd.to_datetime(todo.columns))
    for col, filas in _YF_MAPA.items():
        for f in filas:
            if f in todo.index:
                df[col] = pd.to_numeric(todo.loc[f].values, errors="coerce")
                break
    if "inventory" in df:
        df["inventory"] = df["inventory"].fillna(0.0)
    if "non_current_assets" not in df and {"total_assets", "current_assets"} <= set(df):
        df["non_current_assets"] = df["total_assets"] - df["current_assets"]
    return _limpiar(df)


def _yf_perfil(tk) -> dict:
    try:
        i = tk.info or {}
    except Exception:
        return {}
    return {
        "nombre": i.get("longName") or i.get("shortName"),
        "precio": i.get("currentPrice") or i.get("regularMarketPrice"),
        "market_cap": i.get("marketCap"),
        "beta": i.get("beta"),
        "sector": i.get("sector"),
        "industria": i.get("industry"),
        "descripcion": i.get("longBusinessSummary"),
        "ceo": next((o.get("name") for o in i.get("companyOfficers", []) if "CEO" in (o.get("title") or "")), None),
        "pais": i.get("country"),
        "divisa": i.get("financialCurrency") or i.get("currency"),
        "bolsa": i.get("exchange"),
        "empleados": i.get("fullTimeEmployees"),
        "web": i.get("website"),
        "pct_insiders": i.get("heldPercentInsiders"),
        "pct_instituciones": i.get("heldPercentInstitutions"),
    }


def _yf_precios(tk, anios: int = 11) -> pd.Series:
    try:
        h = tk.history(period=f"{anios}y", auto_adjust=False)
        s = h["Close"].astype(float)
        s.index = s.index.tz_localize(None)
        return s
    except Exception:
        return pd.Series(dtype=float)


def _yf_insiders(tk, datos: DatosEmpresa) -> None:
    try:
        t = tk.insider_transactions
        if t is not None and not t.empty:
            cols = {"Start Date": "Fecha", "Insider": "Persona", "Position": "Cargo",
                    "Transaction": "Operación", "Text": "Detalle", "Shares": "Acciones",
                    "Value": "Importe"}
            t = t[[c for c in cols if c in t]].rename(columns=cols)
            if "Fecha" in t:
                t["Fecha"] = pd.to_datetime(t["Fecha"], errors="coerce")
                t = t.sort_values("Fecha", ascending=False)
            datos.insiders = t
    except Exception:
        datos.avisos.append("No se han podido descargar las operaciones de insiders.")
    try:
        r = tk.insider_purchases
        if r is not None and not r.empty:
            datos.insiders_resumen = r
    except Exception:
        pass
    try:
        mh = tk.major_holders
        if mh is not None and not mh.empty:
            valores = mh.iloc[:, 0].to_dict() if mh.shape[1] == 1 else {}
            if "insidersPercentHeld" in valores:
                datos.accionistas["pct_insiders"] = float(valores["insidersPercentHeld"])
            if "institutionsPercentHeld" in valores:
                datos.accionistas["pct_instituciones"] = float(valores["institutionsPercentHeld"])
    except Exception:
        pass
    try:
        ih = tk.institutional_holders
        if ih is not None and not ih.empty:
            datos.institucionales = ih.head(10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNAS:
        if c not in df:
            df[c] = np.nan
    df = df[COLUMNAS].apply(pd.to_numeric, errors="coerce").sort_index()
    # Un año sin ventas ni beneficio no sirve
    df = df.dropna(subset=["revenue", "net_income"], how="all")
    # El gasto por intereses se trata siempre en positivo
    df["interest_expense"] = df["interest_expense"].abs()
    # El capex se trata siempre en negativo (salida de caja)
    df["capex"] = -df["capex"].abs()
    df["fcf"] = df["fcf"].fillna(df["cfo"] + df["capex"])
    df["total_debt"] = df["total_debt"].fillna(0.0)
    df.index = df.index.year
    df.index.name = "Año"
    df = df[~df.index.duplicated(keep="last")]
    return df


def obtener_datos(ticker: str, clave_fmp: str | None) -> DatosEmpresa:
    ticker = ticker.strip().upper()
    datos = DatosEmpresa(ticker=ticker)

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
    except Exception:
        tk = None

    # 1) Financial Modeling Prep
    if clave_fmp:
        try:
            datos.perfil = _fmp_perfil(ticker, clave_fmp)
            datos.anual = _fmp_anual(ticker, clave_fmp)
            if not datos.anual.empty:
                datos.fuente = "Financial Modeling Prep"
        except Exception as e:
            datos.avisos.append(f"FMP: {e}")
        if datos.fuente:
            try:
                datos.precios = _fmp_precios(ticker, clave_fmp)
            except Exception as e:
                datos.avisos.append(f"FMP (precios): {e}")
            try:
                datos.directivos = _fmp_directivos(ticker, clave_fmp)
            except Exception:
                pass
    else:
        datos.avisos.append("No hay clave de FMP configurada: se usa Yahoo Finance.")

    # 2) Yahoo Finance como respaldo
    if tk is not None:
        perfil_yf = _yf_perfil(tk)
        if not datos.fuente:
            datos.anual = _yf_anual(tk)
            if not datos.anual.empty:
                datos.fuente = "Yahoo Finance"
                if datos.perfil == {} or not datos.perfil.get("nombre"):
                    datos.perfil = perfil_yf
                if clave_fmp:
                    datos.avisos.append(
                        "FMP no tiene este ticker en tu plan; los datos vienen de Yahoo Finance "
                        "(normalmente solo 4-5 años)."
                    )
        # Completar huecos del perfil con Yahoo
        for k, v in perfil_yf.items():
            if datos.perfil.get(k) in (None, "", 0) and v not in (None, ""):
                datos.perfil[k] = v
        if datos.precios.empty:
            datos.precios = _yf_precios(tk)
        _yf_insiders(tk, datos)
        for k in ("pct_insiders", "pct_instituciones"):
            if k not in datos.accionistas and perfil_yf.get(k) is not None:
                datos.accionistas[k] = float(perfil_yf[k])
        if datos.directivos.empty:
            try:
                oficiales = (tk.info or {}).get("companyOfficers", [])
                if oficiales:
                    df = pd.DataFrame(oficiales)
                    cols = {"name": "Nombre", "title": "Cargo", "totalPay": "Retribución",
                            "yearBorn": "Año nac."}
                    datos.directivos = df[[c for c in cols if c in df]].rename(columns=cols)
            except Exception:
                pass

    return datos


def precios_indice(simbolo: str, clave_fmp: str | None) -> pd.Series:
    """Serie de precios del índice de referencia (ETF) para calcular el alfa."""
    if clave_fmp:
        try:
            s = _fmp_precios(simbolo, clave_fmp, anios=6)
            if not s.empty:
                return s
        except Exception:
            pass
    try:
        import yfinance as yf
        return _yf_precios(yf.Ticker(simbolo), anios=6)
    except Exception:
        return pd.Series(dtype=float)
