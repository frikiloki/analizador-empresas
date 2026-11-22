"""Cálculo de los 20 puntos del análisis cuantitativo de la tabla de valoración.

Cada comprobación devuelve un `Resultado` con un estado:
    "ok"    -> cumple el ideal (verde)
    "aviso" -> en zona intermedia o a vigilar (ámbar)
    "mal"   -> no cumple (rojo)
    "info"  -> dato informativo o sin datos suficientes (gris)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datos import DatosEmpresa

OK, AVISO, MAL, INFO = "ok", "aviso", "mal", "info"


@dataclass
class Resultado:
    punto: str
    nombre: str
    valor: str
    ideal: str
    estado: str
    comentario: str = ""


@dataclass
class Parametros:
    tasa_libre_riesgo: float = 0.043
    prima_mercado: float = 0.055
    per_mercado: float = 16.0
    ev_mercado: float = 16.5  # EV/FCF y EV/EBITDA de mercado (16-17x)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _div(a, b):
    try:
        if b is None or a is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def _pct(x, dec=1):
    return "—" if pd.isna(x) else f"{x * 100:.{dec}f} %".replace(".", ",")


def _num(x, dec=2, suf=""):
    return "—" if pd.isna(x) else f"{x:,.{dec}f}{suf}".replace(",", "X").replace(".", ",").replace("X", ".")


def _veces(x):
    return _num(x, 1, "x")


def cagr(serie: pd.Series) -> float:
    s = serie.dropna()
    if len(s) < 2:
        return np.nan
    ini, fin = float(s.iloc[0]), float(s.iloc[-1])
    n = s.index[-1] - s.index[0]
    if ini <= 0 or fin <= 0 or n <= 0:
        return np.nan
    return (fin / ini) ** (1 / n) - 1


def ultimos(serie: pd.Series, n: int) -> pd.Series:
    return serie.dropna().iloc[-n:]


def _cero(x):
    return 0.0 if x is None or pd.isna(x) else x


def _ultimo(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def _tasa_impositiva(df: pd.DataFrame) -> float:
    t = _div(_ultimo(df, "income_tax"), _ultimo(df, "pretax_income"))
    if pd.isna(t) or t < 0 or t > 0.4:
        return 0.21
    return t


def _sin_datos(punto, nombre, ideal, motivo="No hay datos suficientes."):
    return Resultado(punto, nombre, "—", ideal, INFO, motivo)


# ---------------------------------------------------------------------------
# Métricas derivadas (se usan en varios puntos y en el gráfico)
# ---------------------------------------------------------------------------

def precio_fin_de_anio(datos: DatosEmpresa) -> pd.Series:
    p = datos.precios.dropna()
    if p.empty:
        return pd.Series(dtype=float)
    anual = p.groupby(p.index.year).last()
    return anual


def historico_multiplos(datos: DatosEmpresa) -> pd.DataFrame:
    """PER, EV/EBITDA, EV/EBIT, EV/FCF, EV/Ventas de cada año con el precio a cierre de año."""
    df = datos.anual
    precios = precio_fin_de_anio(datos)
    filas = []
    for anio in df.index:
        if anio not in precios.index:
            continue
        p = precios[anio]
        f = df.loc[anio]
        mc = p * f["shares_diluted"] if pd.notna(f["shares_diluted"]) else np.nan
        ev = mc + f["total_debt"] - (f["cash"] if pd.notna(f["cash"]) else 0)
        filas.append({
            "Año": anio,
            "Precio": p,
            "PER": _div(p, f["eps_diluted"]) if f["eps_diluted"] > 0 else np.nan,
            "EV/EBITDA": _div(ev, f["ebitda"]) if f["ebitda"] > 0 else np.nan,
            "EV/EBIT": _div(ev, f["ebit"]) if f["ebit"] > 0 else np.nan,
            "EV/FCF": _div(ev, f["fcf"]) if f["fcf"] > 0 else np.nan,
            "EV/Ventas": _div(ev, f["revenue"]),
        })
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).set_index("Año")


def multiplos_actuales(datos: DatosEmpresa) -> dict:
    df = datos.anual
    precio = datos.precio
    mc = datos.market_cap
    deuda = _ultimo(df, "total_debt")
    caja = _ultimo(df, "cash")
    ev = mc + (deuda if pd.notna(deuda) else 0) - (caja if pd.notna(caja) else 0) if mc else np.nan
    eps = _ultimo(df, "eps_diluted")
    ebitda, ebit, fcf, ventas = (_ultimo(df, c) for c in ("ebitda", "ebit", "fcf", "revenue"))
    return {
        "EV": ev,
        "PER": _div(precio, eps) if eps and eps > 0 else np.nan,
        "EV/EBITDA": _div(ev, ebitda) if ebitda and ebitda > 0 else np.nan,
        "EV/EBIT": _div(ev, ebit) if ebit and ebit > 0 else np.nan,
        "EV/FCF": _div(ev, fcf) if fcf and fcf > 0 else np.nan,
        "EV/Ventas": _div(ev, ventas),
    }


def roic_serie(df: pd.DataFrame) -> pd.Series:
    t = _tasa_impositiva(df)
    capital = df["equity"] + df["total_debt"] - df["cash"].fillna(0)
    capital = capital.where(capital > 0)
    return df["ebit"] * (1 - t) / capital


def wacc(datos: DatosEmpresa, p: Parametros) -> tuple[float, dict]:
    df = datos.anual
    beta = datos.perfil.get("beta")
    beta = float(beta) if beta not in (None, "") and pd.notna(beta) else 1.0
    ke = p.tasa_libre_riesgo + beta * p.prima_mercado
    deuda = _ultimo(df, "total_debt")
    t = _tasa_impositiva(df)
    kd = _div(_ultimo(df, "interest_expense"), deuda)
    if pd.isna(kd) or kd > 0.15:
        kd = p.tasa_libre_riesgo + 0.015
    e = datos.market_cap or np.nan
    d = deuda if pd.notna(deuda) else 0.0
    if pd.isna(e) or e + d == 0:
        return np.nan, {}
    w = e / (e + d) * ke + d / (e + d) * kd * (1 - t)
    return w, {"beta": beta, "ke": ke, "kd": kd, "t": t, "peso_deuda": d / (e + d)}


def alfa(datos: DatosEmpresa, indice: pd.Series, anios: int = 5) -> tuple[float, float, float, float]:
    """Devuelve (diferencia anual, rentab. anual acción, rentab. anual índice, alfa como cociente).

    El alfa como cociente es lo que ha crecido la acción entre lo que ha crecido el índice
    en el periodo (solo precio): > 1 bate al índice, < 1 lo hace peor."""
    p = datos.precios.dropna()
    i = indice.dropna()
    if p.empty or i.empty:
        return np.nan, np.nan, np.nan, np.nan
    desde = max(p.index[-1] - pd.DateOffset(years=anios), p.index[0], i.index[0])
    p, i = p[p.index >= desde], i[i.index >= desde]
    if len(p) < 50 or len(i) < 50:
        return np.nan, np.nan, np.nan, np.nan
    i = i[i.index <= p.index[-1]]
    n = (p.index[-1] - p.index[0]).days / 365.25
    crec_a, crec_i = p.iloc[-1] / p.iloc[0], i.iloc[-1] / i.iloc[0]
    ra, ri = crec_a ** (1 / n) - 1, crec_i ** (1 / n) - 1
    return ra - ri, ra, ri, crec_a / crec_i


# ---------------------------------------------------------------------------
# Los 20 puntos
# ---------------------------------------------------------------------------

def _crecimiento(df, col, nombre) -> Resultado:
    s = ultimos(df[col], 10)
    ideal = "Creciente durante ≥ 5 años"
    if len(s) < 3:
        return _sin_datos("1", nombre, ideal)
    subidas = int((s.diff().dropna() > 0).sum())
    anios = len(s) - 1
    c = cagr(s)
    valor = f"CAGR {_pct(c)} · sube {subidas}/{anios} años"
    if pd.isna(c):
        estado = MAL if s.iloc[-1] <= s.iloc[0] else AVISO
        com = "Hay valores negativos o a cero en el periodo."
    elif c > 0 and subidas >= anios - 1:
        estado, com = OK, "Crecimiento constante."
    elif c > 0:
        estado, com = AVISO, "Crece en el conjunto del periodo, pero con varios años de caída."
    else:
        estado, com = MAL, "No crece en el periodo."
    if len(s) < 6:
        com += f" Solo hay {len(s)} años de datos."
    return Resultado("1", nombre, valor, ideal, estado, com)


def analizar(datos: DatosEmpresa, indice: pd.Series, p: Parametros | None = None) -> list[Resultado]:
    p = p or Parametros()
    df = datos.anual
    r: list[Resultado] = []
    act = multiplos_actuales(datos)
    hist = historico_multiplos(datos)

    # 1. Crecimiento de las magnitudes principales
    for col, nombre in [("operating_income", "Operating income"), ("revenue", "Total revenues"),
                        ("net_income", "Net income"), ("eps_diluted", "EPS diluted"),
                        ("equity", "Equity (patrimonio neto)")]:
        r.append(_crecimiento(df, col, nombre))

    # 2. Acciones diluidas en los últimos 5 años
    s = ultimos(df["shares_diluted"], 6)
    ideal = "No aumentan en 5 años (≤)"
    if len(s) >= 2:
        var = s.iloc[-1] / s.iloc[0] - 1
        est = OK if var <= 0 else (AVISO if var <= 0.05 else MAL)
        com = "Recompra acciones." if var < 0 else ("Ligera dilución." if var <= 0.05 else "Dilución relevante para el accionista.")
        r.append(Resultado("2", "Acciones diluidas", f"{_pct(var)} en {s.index[-1] - s.index[0]} años", ideal, est, com))
    else:
        r.append(_sin_datos("2", "Acciones diluidas", ideal))

    # 3. Márgenes
    roe = _div(_ultimo(df, "net_income"), _ultimo(df, "equity"))
    roe_med = (df["net_income"] / df["equity"]).iloc[-5:].median()
    est = INFO if pd.isna(roe) else (OK if roe >= 0.15 else (AVISO if roe >= 0.10 else MAL))
    r.append(Resultado("3", "ROE", f"{_pct(roe)} (mediana 5 a: {_pct(roe_med)})", "≥ 15 %", est,
                       "Compara con el sector y competidores en la pestaña Competidores."))
    mn = _div(_ultimo(df, "net_income"), _ultimo(df, "revenue"))
    mn_med = (df["net_income"] / df["revenue"]).iloc[-5:].median()
    est = INFO if pd.isna(mn) else (OK if mn >= 0.15 else (AVISO if mn >= 0.10 else MAL))
    r.append(Resultado("3", "Margen neto", f"{_pct(mn)} (mediana 5 a: {_pct(mn_med)})", "≥ 15 %", est,
                       "Compara con el sector y competidores en la pestaña Competidores."))

    # 4. PEG
    crec_eps = cagr(ultimos(df["eps_diluted"], 6))
    per = act["PER"]
    peg = _div(per, crec_eps * 100) if pd.notna(crec_eps) and crec_eps > 0 else np.nan
    if pd.isna(peg):
        r.append(_sin_datos("4", "PEG", "0 – 1", "No calculable: el BPA no crece o el PER es negativo."))
    else:
        est = OK if 0 < peg <= 1 else (AVISO if peg <= 2 else MAL)
        r.append(Resultado("4", "PEG", _num(peg), "0 – 1", est,
                           f"PER {_veces(per)} / crecimiento BPA {_pct(crec_eps)} anual (últimos 5 años)."))

    # 5. ROIC > WACC
    roic = roic_serie(df)
    roic_ult = roic.dropna().iloc[-1] if not roic.dropna().empty else np.nan
    w, det = wacc(datos, p)
    if pd.isna(roic_ult) or pd.isna(w):
        r.append(_sin_datos("5", "ROIC > WACC", "ROIC > WACC"))
    else:
        dif = roic_ult - w
        est = OK if dif > 0.02 else (AVISO if dif > 0 else MAL)
        r.append(Resultado("5", "ROIC > WACC", f"ROIC {_pct(roic_ult)} vs WACC {_pct(w)}", "ROIC > WACC", est,
                           f"WACC estimado con beta {_num(det['beta'])}, coste del capital {_pct(det['ke'])}, "
                           f"coste de la deuda {_pct(det['kd'])} y {_pct(det['peso_deuda'], 0)} de deuda."))

    # 6. PER y EV/EBITDA frente a su mediana histórica
    for m in ("PER", "EV/EBITDA"):
        med = hist[m].median() if not hist.empty and m in hist else np.nan
        actual = act[m]
        n = int(hist[m].notna().sum()) if not hist.empty and m in hist else 0
        if pd.isna(med) or pd.isna(actual):
            r.append(_sin_datos("6", f"{m} vs mediana histórica", "Por debajo de su mediana"))
            continue
        rel = actual / med - 1
        est = OK if rel <= 0 else (AVISO if rel <= 0.2 else MAL)
        com = (f"Cotiza un {_pct(-rel, 0)} por debajo de su mediana: revisa si el problema es coyuntural o estructural."
               if rel <= 0 else f"Cotiza un {_pct(rel, 0)} por encima de su mediana.")
        r.append(Resultado("6", f"{m} vs mediana {n} años", f"{_veces(actual)} vs {_veces(med)}",
                           "Por debajo de su mediana", est, com))

    # 7. Precio frente a BPA (detalle en el gráfico)
    per_med = hist["PER"].median() if not hist.empty else np.nan
    eps_ult = _ultimo(df, "eps_diluted")
    precio = datos.precio
    if pd.notna(per_med) and pd.notna(eps_ult) and eps_ult > 0 and precio:
        justo = eps_ult * per_med
        desc = precio / justo - 1
        est = OK if desc <= -0.15 else (AVISO if desc <= 0.10 else MAL)
        r.append(Resultado("7", "Precio vs BPA × PER mediano", f"Precio {_num(precio)} vs {_num(justo)}",
                           "Precio muy por debajo de la línea de BPA", est,
                           f"El precio está un {_pct(abs(desc), 0)} {'por debajo' if desc < 0 else 'por encima'} "
                           f"de BPA × PER mediano ({_veces(per_med)}). Ver gráfico."))
    else:
        r.append(_sin_datos("7", "Precio vs BPA", "Precio muy por debajo de la línea de BPA"))

    # 8. Endeudamiento
    deuda_neta = _ultimo(df, "total_debt") - (_ultimo(df, "cash") if pd.notna(_ultimo(df, "cash")) else 0)
    nd_ebitda = _div(deuda_neta, _ultimo(df, "ebitda"))
    if pd.isna(nd_ebitda):
        r.append(_sin_datos("8", "Deuda neta / EBITDA", "< 2"))
    else:
        if deuda_neta < 0:
            est, com = OK, "Tiene más caja que deuda: sin riesgo de quiebra."
        elif nd_ebitda < 2:
            est, com = OK, "Buena salud financiera."
        elif nd_ebitda <= 3:
            est, com = AVISO, "Zona intermedia (2-3)."
        else:
            est, com = MAL, "Posibles problemas financieros."
        r.append(Resultado("8", "Deuda neta / EBITDA", _veces(nd_ebitda), "< 0 sin riesgo · < 2 buena · > 3 problemas", est,
                           com + " (Tu tabla dice «interest expenses»; por los umbrales se ha usado EBITDA.)"))
    nd_eq = _div(deuda_neta, _ultimo(df, "equity"))
    if pd.isna(nd_eq):
        r.append(_sin_datos("8", "Deuda neta / Equity", "≤ 1"))
    else:
        if _ultimo(df, "equity") < 0:
            est, com = MAL, "Patrimonio neto negativo."
        elif nd_eq < 0.3:
            est, com = OK, "Administración conservadora."
        elif nd_eq <= 1:
            est, com = OK, "Equilibrado."
        elif nd_eq <= 2:
            est, com = AVISO, "Por encima de 1: vigilar."
        else:
            est, com = MAL, "Deuda preocupante."
        r.append(Resultado("8", "Deuda neta / Equity", _num(nd_eq), "< 0,3 conservador · 1 equilibrado · > 2 preocupante", est, com))
    d_act = _div(_ultimo(df, "total_debt"), _ultimo(df, "total_assets"))
    if pd.isna(d_act):
        r.append(_sin_datos("8", "Deuda / Activos totales", "< 40 %"))
    else:
        est = OK if d_act < 0.40 else (AVISO if d_act < 0.50 else MAL)
        r.append(Resultado("8", "Deuda / Activos totales", _pct(d_act), "< 40 %", est))

    # 9. Cobertura de intereses
    cob = _div(_ultimo(df, "ebit"), _ultimo(df, "interest_expense"))
    if _ultimo(df, "interest_expense") in (0, 0.0) or (pd.isna(cob) and _ultimo(df, "total_debt") == 0):
        r.append(Resultado("9", "EBIT / Gasto por intereses", "Sin intereses", "> 10x", OK, "No paga intereses."))
    elif pd.isna(cob):
        r.append(_sin_datos("9", "EBIT / Gasto por intereses", "> 10x"))
    else:
        est = OK if cob > 10 else (AVISO if cob > 5 else MAL)
        r.append(Resultado("9", "EBIT / Gasto por intereses", _veces(cob), "> 10x", est,
                           "Hace frente a los intereses con holgura." if cob > 10 else ""))

    # 10. Current ratio
    cr = _div(_ultimo(df, "current_assets"), _ultimo(df, "current_liabilities"))
    if pd.isna(cr):
        r.append(_sin_datos("10", "Current ratio", "> 1"))
    else:
        r.append(Resultado("10", "Current ratio", _num(cr), "> 1", OK if cr > 1 else MAL,
                           "Puede pagar a corto plazo." if cr > 1 else "Activo corriente menor que pasivo corriente."))

    # 11. CAGR BPA y EBIT frente a CAGR ventas
    c_rev = cagr(ultimos(df["revenue"], 6))
    var_acc = (lambda s: s.iloc[-1] / s.iloc[0] - 1 if len(s) >= 2 else np.nan)(ultimos(df["shares_diluted"], 6))
    for col, nombre in [("eps_diluted", "CAGR EPS > CAGR ventas"), ("ebit", "CAGR EBIT > CAGR ventas")]:
        c = cagr(ultimos(df[col], 6))
        if pd.isna(c) or pd.isna(c_rev):
            r.append(_sin_datos("11", nombre, "Mayor que el de las ventas"))
            continue
        est = OK if c > c_rev else MAL
        com = ""
        if col == "eps_diluted" and c > c_rev and pd.notna(var_acc) and var_acc < 0:
            com = f"Muy eficiente: ayudado por la reducción de acciones ({_pct(var_acc)})."
        r.append(Resultado("11", nombre, f"{_pct(c)} vs {_pct(c_rev)}", "Mayor que el de las ventas", est, com))

    # 12. Acid ratio
    acid = _div(_ultimo(df, "current_assets") - _cero(_ultimo(df, "inventory")), _ultimo(df, "current_liabilities"))
    if pd.isna(acid):
        r.append(_sin_datos("12", "Acid ratio (prueba ácida)", "≥ 1"))
    else:
        sector = (datos.perfil.get("sector") or "") + " " + (datos.perfil.get("industria") or "")
        retail = any(x in sector.lower() for x in ("retail", "consumer", "grocery", "discount"))
        est = OK if acid >= 1 else (AVISO if retail else MAL)
        com = "" if acid >= 1 else ("Bajo, pero en retail es habitual." if retail else "Preocupante.")
        r.append(Resultado("12", "Acid ratio (prueba ácida)", _num(acid), "≥ 1 (< 1 preocupante salvo retail)", est, com))

    # 13. Múltiplos de valoración
    refs = {"PER": p.per_mercado, "EV/FCF": p.ev_mercado, "EV/EBITDA": p.ev_mercado}
    for m in ("PER", "EV/FCF", "EV/EBITDA", "EV/EBIT", "EV/Ventas"):
        actual = act[m]
        med = hist[m].median() if not hist.empty and m in hist else np.nan
        texto_med = f" · mediana hist. {_veces(med)}" if pd.notna(med) else ""
        if pd.isna(actual):
            r.append(_sin_datos("13", m, "Por debajo de mercado / su histórico", "No calculable (denominador negativo o sin datos)."))
            continue
        if m in refs:
            ref = refs[m]
            est = OK if actual <= ref else (AVISO if actual <= ref * 1.25 else MAL)
            r.append(Resultado("13", m, _veces(actual) + texto_med, f"Mercado ≈ {_veces(ref)}", est,
                               "Compara también con el sector y los competidores."))
        else:
            est = INFO if pd.isna(med) else (OK if actual <= med else AVISO)
            r.append(Resultado("13", m, _veces(actual) + texto_med, "Comparar con histórico, sector y competidor", est))

    # 14. Conversión EBITDA -> FCF
    conv = _div(_ultimo(df, "fcf"), _ultimo(df, "ebitda"))
    conv_med = (df["fcf"] / df["ebitda"]).iloc[-5:].median()
    if pd.isna(conv):
        r.append(_sin_datos("14", "Conversión EBITDA → FCF", "> 60 %"))
    else:
        est = OK if conv > 0.6 else (AVISO if conv > 0.4 else MAL)
        r.append(Resultado("14", "Conversión EBITDA → FCF", f"{_pct(conv)} (mediana 5 a: {_pct(conv_med)})", "> 60 %", est))

    # 15. FCF creciente y capital circulante
    fcf = ultimos(df["fcf"], 6)
    if len(fcf) >= 3:
        c = cagr(fcf)
        subidas = int((fcf.diff().dropna() > 0).sum())
        if pd.notna(c) and c > 0:
            est = OK if subidas >= len(fcf) - 2 else AVISO
        else:
            est = MAL
        r.append(Resultado("15", "FCF creciente", f"CAGR {_pct(c)} · sube {subidas}/{len(fcf) - 1} años", "Creciente", est))
    else:
        r.append(_sin_datos("15", "FCF creciente", "Creciente"))
    wc = ultimos(df["change_wc"], 5)
    if not wc.empty:
        r.append(Resultado("15", "Variación del capital circulante (WC)",
                           " · ".join(f"{a}: {_num(v / 1e6, 0)} M" for a, v in wc.items()), "Estable", INFO,
                           "Valores negativos = el circulante consume caja."))

    # 16. Alfa frente al índice
    a, ra, ri, alfa_ratio = alfa(datos, indice)
    if pd.isna(alfa_ratio):
        r.append(_sin_datos("16", "Alfa vs índice (5 años)", "> 1 bate al índice · < 1 peor"))
    else:
        r.append(Resultado("16", "Alfa vs índice (5 años)", _num(alfa_ratio), "> 1 bate al índice · < 1 peor",
                           OK if alfa_ratio > 1 else MAL,
                           "Bate al índice." if alfa_ratio > 1 else "Peor que el índice."))

    # 17. ROIC
    if pd.isna(roic_ult):
        r.append(_sin_datos("17", "ROIC", "> 15 %"))
    else:
        if roic_ult > 0.20:
            est, com = OK, "Excelente."
        elif roic_ult > 0.15:
            est, com = OK, "Muy bueno."
        elif roic_ult >= 0.10:
            est, com = AVISO, "Normal (mercado ≈ 13 %)."
        else:
            est, com = MAL, "No es eficiente."
        roic_med = roic.dropna().iloc[-5:].median()
        r.append(Resultado("17", "ROIC", f"{_pct(roic_ult)} (mediana 5 a: {_pct(roic_med)})",
                           "< 10 % malo · 10-15 % normal · > 15 % muy bueno · > 20 % excelente", est, com))

    # 18. ROE frente a ROA
    roa = _div(_ultimo(df, "net_income"), _ultimo(df, "total_assets"))
    if pd.isna(roe) or pd.isna(roa):
        r.append(_sin_datos("18", "ROE vs ROA", "ROE ≈ ROA"))
    else:
        rel = _div(roe, roa)
        if roe < 0:
            est, com = MAL, "Los accionistas pierden dinero."
        elif rel <= 1.5:
            est, com = OK, "ROE cercano al ROA: rentabilidad sin depender de la deuda."
        elif rel <= 3:
            est, com = AVISO, "Parte del ROE viene del apalancamiento."
        else:
            est, com = AVISO, "El ROE se apoya mucho en la deuda (o en patrimonio pequeño por recompras)."
        r.append(Resultado("18", "ROE vs ROA", f"ROE {_pct(roe)} · ROA {_pct(roa)}", "Ideal ROE ≈ ROA", est, com))

    # 19. Solvencia general
    solv = _div(_ultimo(df, "total_assets"), _ultimo(df, "total_liabilities"))
    if pd.isna(solv):
        r.append(_sin_datos("19", "Solvencia general (activo / pasivo total)", "> 2"))
    else:
        est = OK if solv > 2 else (AVISO if solv >= 1.5 else MAL)
        com = {OK: "Buena.", AVISO: "Ajustada.", MAL: "Deficiente."}[est]
        r.append(Resultado("19", "Solvencia general (activo / pasivo total)", _num(solv),
                           "< 1,5 deficiente · 1,5-2 ajustado · > 2 bueno", est, com))

    # 20. Estabilidad
    eq = _ultimo(df, "equity")
    est_r = _div(_ultimo(df, "non_current_assets"), eq)
    if pd.isna(est_r):
        r.append(_sin_datos("20", "Estabilidad (activo fijo / recursos propios)", "≤ 1"))
    else:
        if eq <= 0:
            est, com = MAL, "Patrimonio neto negativo."
        elif est_r < 0.95:
            est, com = OK, "El inmovilizado se financia con recursos a largo plazo."
        elif est_r <= 1.05:
            est, com = OK, "Todos los recursos propios están en el activo fijo."
        elif est_r <= 2:
            est, com = AVISO, "Parte del inmovilizado se financia a corto plazo o con deuda."
        else:
            est, com = MAL, "Muy por encima de 1: riesgo de tensiones de liquidez."
        r.append(Resultado("20", "Estabilidad (activo fijo / recursos propios)", _num(est_r),
                           "< 1 bien · = 1 · > 1 parte a corto · ≫ 1 riesgo", est, com))

    return r


def resumen(resultados: list[Resultado]) -> dict:
    c = {OK: 0, AVISO: 0, MAL: 0, INFO: 0}
    for x in resultados:
        c[x.estado] += 1
    evaluados = c[OK] + c[AVISO] + c[MAL]
    c["puntuacion"] = (c[OK] + 0.5 * c[AVISO]) / evaluados if evaluados else np.nan
    return c


def tabla_comparativa(datos: DatosEmpresa) -> dict:
    """Métricas clave de una empresa para compararla con sus competidores."""
    df = datos.anual
    act = multiplos_actuales(datos)
    roic = roic_serie(df).dropna()
    return {
        "Empresa": datos.nombre,
        "Ventas (M)": _ultimo(df, "revenue") / 1e6,
        "CAGR ventas 5a": cagr(ultimos(df["revenue"], 6)),
        "Margen operativo": _div(_ultimo(df, "operating_income"), _ultimo(df, "revenue")),
        "Margen neto": _div(_ultimo(df, "net_income"), _ultimo(df, "revenue")),
        "ROE": _div(_ultimo(df, "net_income"), _ultimo(df, "equity")),
        "ROIC": roic.iloc[-1] if not roic.empty else np.nan,
        "I+D / ventas": _div(_ultimo(df, "rnd"), _ultimo(df, "revenue")),
        "Deuda neta/EBITDA": _div(_ultimo(df, "total_debt") - _cero(_ultimo(df, "cash")), _ultimo(df, "ebitda")),
        "PER": act["PER"],
        "EV/EBITDA": act["EV/EBITDA"],
        "EV/FCF": act["EV/FCF"],
    }
