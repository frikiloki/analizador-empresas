"""Analizador de empresas según la tabla de valoración propia.

Ejecutar en local:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador import cualitativo as cual
from analizador import ratios as rt
from analizador.datos import obtener_datos, precios_indice

st.set_page_config(page_title="Analizador de empresas", page_icon="📊", layout="wide")

COLORES = {"ok": "#d4edda", "aviso": "#fff3cd", "mal": "#f8d7da", "info": "#eceff1"}
ICONOS = {"ok": "✅", "aviso": "🟡", "mal": "❌", "info": "⚪"}
INDICES = {
    "S&P 500 (SPY)": "SPY",
    "MSCI World (URTH)": "URTH",
    "Euro Stoxx 50 (FEZ)": "FEZ",
    "Nasdaq 100 (QQQ)": "QQQ",
}


def secreto(nombre: str) -> str | None:
    try:
        v = st.secrets.get(nombre)
        return str(v) if v else None
    except Exception:
        return None


CLAVE_FMP = secreto("FMP_API_KEY")
CLAVE_IA = secreto("ANTHROPIC_API_KEY")
MODELO_IA = secreto("ANTHROPIC_MODEL") or "claude-sonnet-5-5"


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cargar(ticker: str):
    return obtener_datos(ticker, CLAVE_FMP)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cargar_indice(simbolo: str):
    return precios_indice(simbolo, CLAVE_FMP)


def fmt_millones(x, divisa=""):
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1e9:
        return f"{x / 1e9:,.1f} mil M {divisa}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x / 1e6:,.0f} M {divisa}".replace(",", ".")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Analizador")
    with st.form("busqueda"):
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", ""),
                               placeholder="Ej.: AAPL, MSFT, ITX.MC").strip().upper()
        competidores = st.text_input("Competidores (opcional)", placeholder="Ej.: GOOGL, META",
                                     help="Tickers separados por comas para la pestaña Competidores.")
        enviar = st.form_submit_button("Analizar", type="primary", use_container_width=True)
    if enviar and ticker:
        st.session_state["ticker"] = ticker
        st.session_state["competidores"] = competidores

    indice_nombre = st.selectbox("Índice para el alfa", list(INDICES))
    with st.expander("Parámetros"):
        rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 10.0, 4.3, 0.1) / 100
        erp = st.number_input("Prima de riesgo de mercado (%)", 0.0, 12.0, 5.5, 0.1) / 100
        per_m = st.number_input("PER de mercado", 5.0, 40.0, 16.0, 0.5)
        ev_m = st.number_input("EV/FCF y EV/EBITDA de mercado", 5.0, 40.0, 16.5, 0.5)
    st.caption(
        ("🟢 FMP conectado" if CLAVE_FMP else "⚪ Sin clave de FMP (se usa Yahoo Finance)")
        + "  \n" + ("🟢 IA disponible" if CLAVE_IA else "⚪ IA no configurada (opcional)")
    )

ticker = st.session_state.get("ticker")
if not ticker:
    st.title("Analizador de empresas")
    st.write("Escribe un ticker en la barra lateral y pulsa **Analizar**. La app revisa la empresa "
             "con los 20 puntos cuantitativos de tu tabla de valoración y prepara la parte cualitativa.")
    st.info("Para acciones fuera de EE. UU. usa el sufijo de Yahoo Finance: `.MC` Madrid, `.PA` París, "
            "`.DE` Fráncfort, `.L` Londres, `.AS` Ámsterdam, `.MI` Milán…")
    st.stop()

with st.spinner(f"Descargando datos de {ticker}…"):
    datos = cargar(ticker)
    indice = cargar_indice(INDICES[indice_nombre])

if datos.anual.empty:
    st.error(f"No he encontrado datos financieros para **{ticker}**. Revisa el ticker "
             "(fuera de EE. UU. añade el sufijo de la bolsa, p. ej. `ITX.MC`).")
    for a in datos.avisos:
        st.caption(a)
    st.stop()

params = rt.Parametros(tasa_libre_riesgo=rf, prima_mercado=erp, per_mercado=per_m, ev_mercado=ev_m)
resultados = rt.analizar(datos, indice, params)
res = rt.resumen(resultados)
divisa = datos.perfil.get("divisa") or ""

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.title(f"{datos.nombre} ({datos.ticker})")
st.caption(" · ".join(str(x) for x in [datos.perfil.get("sector"), datos.perfil.get("industria"),
                                        datos.perfil.get("pais"), datos.perfil.get("bolsa"),
                                        f"Datos: {datos.fuente}"] if x))
c = st.columns(5)
c[0].metric("Precio", f"{datos.precio:,.2f} {divisa}" if datos.precio else "—")
c[1].metric("Capitalización", fmt_millones(datos.market_cap, divisa))
c[2].metric("✅ Cumple", res["ok"])
c[3].metric("🟡 A vigilar", res["aviso"])
c[4].metric("❌ No cumple", res["mal"])
if pd.notna(res["puntuacion"]):
    st.progress(float(res["puntuacion"]), text=f"Puntuación cuantitativa: {res['puntuacion'] * 100:.0f} / 100 "
                                                "(cumple = 1, a vigilar = 0,5)")
for a in datos.avisos:
    st.caption(f"ℹ️ {a}")

tab_cuanti, tab_graf, tab_cuali, tab_comp, tab_datos = st.tabs(
    ["Cuantitativo", "Gráficos", "Cualitativo", "Competidores", "Datos"])

# ---------------------------------------------------------------------------
# Cuantitativo
# ---------------------------------------------------------------------------
with tab_cuanti:
    tabla = pd.DataFrame([{
        "": ICONOS[r.estado], "N.º": r.punto, "Ratio": r.nombre, "Valor": r.valor,
        "Ideal": r.ideal, "Comentario": r.comentario, "_estado": r.estado,
    } for r in resultados])

    filtro = st.radio("Mostrar", ["Todos", "No cumple", "A vigilar", "Cumple"], horizontal=True)
    mapa = {"No cumple": "mal", "A vigilar": "aviso", "Cumple": "ok"}
    if filtro != "Todos":
        tabla = tabla[tabla["_estado"] == mapa[filtro]]

    def _colorear(fila):
        return [f"background-color: {COLORES[fila['_estado']]}"] * len(fila)

    estilo = tabla.style.apply(_colorear, axis=1).hide(axis="index")
    st.dataframe(estilo, use_container_width=True, hide_index=True,
                 column_config={"_estado": None, "Comentario": st.column_config.TextColumn(width="large")},
                 height=min(38 * (len(tabla) + 1), 1200))
    st.caption("Los datos son anuales (último ejercicio cerrado) salvo el precio. El WACC es una estimación: "
               "ajusta los parámetros en la barra lateral. Herramienta de apoyo, no es una recomendación de inversión.")

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
with tab_graf:
    df = datos.anual
    hist = rt.historico_multiplos(datos)

    # 7. Precio vs BPA
    st.subheader("7 · Precio frente a BPA")
    per_med = hist["PER"].median() if not hist.empty else np.nan
    p = datos.precios.dropna()
    if not p.empty and df["eps_diluted"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index, y=p.values, name="Precio", line=dict(color="#1f77b4")))
        eps = df["eps_diluted"].dropna()
        fechas = pd.to_datetime([f"{a}-12-31" for a in eps.index])
        if pd.notna(per_med):
            fig.add_trace(go.Scatter(x=fechas, y=eps.values * per_med, name=f"BPA × PER mediano ({per_med:.1f}x)",
                                     mode="lines+markers", line=dict(color="#2ca02c", shape="hv")))
        fig.add_trace(go.Scatter(x=fechas, y=eps.values * per_m, name=f"BPA × PER mercado ({per_m:.0f}x)",
                                 mode="lines", line=dict(color="#ff7f0e", dash="dot", shape="hv")))
        fig.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h"),
                          margin=dict(t=10, b=10), yaxis_title=divisa)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cuando el precio (azul) queda muy por debajo de la línea de BPA (verde), puede haber oportunidad.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1 · Magnitudes principales")
        fig = go.Figure()
        for c_, n_ in [("revenue", "Ventas"), ("operating_income", "Operating income"), ("net_income", "Beneficio neto"),
                       ("fcf", "FCF")]:
            fig.add_trace(go.Bar(x=df.index, y=df[c_] / 1e6, name=n_))
        fig.update_layout(height=380, barmode="group", legend=dict(orientation="h"), margin=dict(t=10, b=10),
                          yaxis_title=f"M {divisa}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("3 · Márgenes y rentabilidad")
        roic = rt.roic_serie(df)
        fig = go.Figure()
        for serie, n_ in [(df["operating_income"] / df["revenue"], "Margen operativo"),
                          (df["net_income"] / df["revenue"], "Margen neto"),
                          (df["net_income"] / df["equity"], "ROE"), (roic, "ROIC")]:
            fig.add_trace(go.Scatter(x=df.index, y=serie * 100, name=n_, mode="lines+markers"))
        fig.add_hline(y=15, line_dash="dot", line_color="grey", annotation_text="15 %")
        fig.update_layout(height=380, legend=dict(orientation="h"), margin=dict(t=10, b=10), yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("2 · Acciones diluidas")
        fig = go.Figure(go.Bar(x=df.index, y=df["shares_diluted"] / 1e6, marker_color="#9467bd"))
        fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title="Millones")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("6 · PER y EV/EBITDA históricos")
        if not hist.empty:
            fig = go.Figure()
            for m in ("PER", "EV/EBITDA", "EV/FCF"):
                fig.add_trace(go.Scatter(x=hist.index, y=hist[m], name=m, mode="lines+markers"))
            fig.update_layout(height=320, legend=dict(orientation="h"), margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Sin histórico de precios suficiente.")

# ---------------------------------------------------------------------------
# Cualitativo
# ---------------------------------------------------------------------------
with tab_cuali:
    clave_estado = f"cuali_{ticker}"
    if clave_estado not in st.session_state:
        st.session_state[clave_estado] = {"notas": {}, "valoraciones": {}, "ia": {}}
    estado_c = st.session_state[clave_estado]

    st.subheader("Datos de la empresa")
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**1. Modelo de negocio**")
        st.write(datos.perfil.get("descripcion") or "Sin descripción.")
        st.markdown(f"**Ciclicidad (dato):** {cual.ciclicidad(datos)}")
        rnd = (datos.anual["rnd"] / datos.anual["revenue"]).dropna()
        if not rnd.empty:
            st.markdown("**Gasto en I+D / ventas:** " + " · ".join(f"{i}: {v * 100:.1f} %" for i, v in rnd.iloc[-5:].items()))
    with b:
        st.markdown("**4. Equipo directivo · skin in the game**")
        pi = datos.accionistas.get("pct_insiders")
        pinst = datos.accionistas.get("pct_instituciones")
        est_sk, txt_sk = cual.valoracion_skin_in_the_game(pi)
        m1, m2 = st.columns(2)
        m1.metric("% en manos de insiders", f"{pi * 100:.2f} %" if pi is not None else "—")
        m2.metric("% instituciones", f"{pinst * 100:.1f} %" if pinst is not None else "—")
        st.markdown(f"{ICONOS[est_sk]} {txt_sk}")
        if pinst is not None and pi is not None:
            st.caption(f"Retail y otros ≈ {max(0.0, 1 - pi - pinst) * 100:.1f} % (estimación).")

        ins = cual.resumen_insiders(datos)
        st.markdown("**Compras y ventas de insiders (12 meses)**")
        if ins["hay_datos"]:
            i1, i2 = st.columns(2)
            i1.metric("Compras", ins["compras"], help=f"{ins['acc_compradas']:,.0f} acciones")
            i2.metric("Ventas", ins["ventas"], help=f"{ins['acc_vendidas']:,.0f} acciones")
            if ins["ventas"] > 0 and ins["compras"] == 0:
                st.markdown("🟡 Solo ventas. Revisa si son planes programados (10b5-1) o ventas para impuestos.")
            elif ins["compras"] > ins["ventas"]:
                st.markdown("✅ Predominan las compras: señal de confianza de los directivos.")
        else:
            st.caption("Sin datos de operaciones de insiders para este ticker.")

    if not datos.directivos.empty:
        with st.expander("Directivos y retribución"):
            st.dataframe(datos.directivos, use_container_width=True, hide_index=True)
    if not datos.insiders.empty:
        with st.expander("Operaciones de insiders (detalle)"):
            st.dataframe(datos.insiders.head(50), use_container_width=True, hide_index=True)
    if not datos.institucionales.empty:
        with st.expander("Principales accionistas institucionales"):
            st.dataframe(datos.institucionales, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Análisis con IA")
    if CLAVE_IA:
        o1, o2 = st.columns([2, 1])
        buscar = o1.toggle("Buscar noticias recientes en internet", value=True,
                           help="La IA busca noticias, cambios de directivos, operaciones de insiders y competidores. "
                                "Coste orientativo: 0,10-0,15 $ por empresa (sin búsqueda ≈ 0,04 $).")
        max_b = o2.slider("Máx. búsquedas", 1, 10, 5, disabled=not buscar)
        if st.button("Generar análisis cualitativo con IA", type="primary"):
            with st.spinner("Buscando información y analizando… (1-2 minutos)"):
                try:
                    estado_c["ia"] = cual.analisis_ia(datos, resultados, estado_c["notas"], CLAVE_IA,
                                                      MODELO_IA, buscar=buscar, max_busquedas=max_b)
                except Exception as e:
                    if buscar:
                        st.warning("La búsqueda web ha fallado (puede que no esté activada en tu cuenta de "
                                   f"Anthropic: Settings → Privacy en platform.claude.com). Se repite sin búsqueda. "
                                   f"Detalle: {e}")
                        try:
                            estado_c["ia"] = cual.analisis_ia(datos, resultados, estado_c["notas"], CLAVE_IA,
                                                              MODELO_IA, buscar=False)
                        except Exception as e2:
                            st.error(f"No se ha podido generar el análisis: {e2}")
                    else:
                        st.error(f"No se ha podido generar el análisis: {e}")
        ia = estado_c["ia"]
        if isinstance(ia, dict) and ia.get("texto"):
            st.markdown(ia["texto"])
            if ia["fuentes"]:
                with st.expander(f"Fuentes consultadas ({len(ia['fuentes'])})"):
                    for titulo, url in ia["fuentes"]:
                        st.markdown(f"- [{titulo}]({url})")
            st.caption(f"Redactado por IA ({MODELO_IA}) · {ia['busquedas']} búsquedas · coste aprox. "
                       f"{ia['coste']:.3f} $. Contrasta lo importante en las fuentes.")
    else:
        st.caption("Opcional: añade `ANTHROPIC_API_KEY` en los secretos de Streamlit para que la IA redacte "
                   "el análisis de entorno, equipo directivo y moat siguiendo tu tabla.")

    st.divider()
    st.subheader("Tu checklist")
    st.caption("Valora cada punto y añade notas. Se guarda mientras la pestaña esté abierta: descarga el informe al terminar.")
    opciones = ["—", "✅ Favorable", "🟡 Neutral", "❌ Desfavorable"]
    for bloque, items in cual.CHECKLIST.items():
        with st.expander(bloque):
            for it in items:
                k = f"{bloque}|{it}"
                c1, c2 = st.columns([2, 3])
                actual = estado_c["valoraciones"].get(k, "—")
                estado_c["valoraciones"][k] = c1.selectbox(it, opciones, index=opciones.index(actual),
                                                           key=f"v_{ticker}_{k}")
                estado_c["notas"][k] = c2.text_input("Notas", value=estado_c["notas"].get(k, ""),
                                                     key=f"n_{ticker}_{k}", label_visibility="hidden")

    ia = estado_c["ia"] if isinstance(estado_c["ia"], dict) else {}
    texto_ia = ia.get("texto", "")
    if ia.get("fuentes"):
        texto_ia += "\n\n### Fuentes\n" + "\n".join(f"- [{t}]({u})" for t, u in ia["fuentes"])
    informe = cual.exportar_markdown(datos, resultados, estado_c["notas"], estado_c["valoraciones"], texto_ia)
    st.download_button("⬇️ Descargar informe (Markdown)", informe, file_name=f"analisis_{ticker}.md",
                       mime="text/markdown")

# ---------------------------------------------------------------------------
# Competidores
# ---------------------------------------------------------------------------
with tab_comp:
    lista = [t.strip().upper() for t in st.session_state.get("competidores", "").split(",") if t.strip()]
    if not lista:
        st.info("Escribe los tickers de los competidores en la barra lateral para comparar márgenes, "
                "rentabilidad, deuda y múltiplos (puntos 3, 13 y entorno micro).")
    else:
        filas = [rt.tabla_comparativa(datos)]
        with st.spinner("Descargando competidores…"):
            for t in lista[:6]:
                d = cargar(t)
                if d.anual.empty:
                    st.warning(f"Sin datos para {t}.")
                    continue
                filas.append(rt.tabla_comparativa(d))
        comp = pd.DataFrame(filas).set_index("Empresa")
        pct = ["CAGR ventas 5a", "Margen operativo", "Margen neto", "ROE", "ROIC", "I+D / ventas"]
        formato = {c_: "{:.1%}" for c_ in pct}
        formato.update({"Ventas (M)": "{:,.0f}", "Deuda neta/EBITDA": "{:.2f}",
                        "PER": "{:.1f}x", "EV/EBITDA": "{:.1f}x", "EV/FCF": "{:.1f}x"})
        st.dataframe(comp.style.format(formato, na_rep="—"), use_container_width=True)
        st.caption("Datos del último ejercicio de cada empresa. Las ventas están en la divisa de cada una.")

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
with tab_datos:
    st.caption(f"Fuente: {datos.fuente}. Cifras en {divisa}.")
    nombres = {
        "revenue": "Ventas", "operating_income": "Operating income", "ebit": "EBIT", "ebitda": "EBITDA",
        "net_income": "Beneficio neto", "eps_diluted": "BPA diluido", "shares_diluted": "Acciones diluidas",
        "interest_expense": "Gasto por intereses", "rnd": "I+D", "income_tax": "Impuestos",
        "pretax_income": "Beneficio antes de impuestos", "total_assets": "Activo total",
        "total_liabilities": "Pasivo total", "total_debt": "Deuda total", "cash": "Caja",
        "equity": "Patrimonio neto", "current_assets": "Activo corriente",
        "current_liabilities": "Pasivo corriente", "inventory": "Inventario",
        "non_current_assets": "Activo no corriente", "cfo": "Flujo de caja operativo",
        "capex": "Capex", "fcf": "FCF", "change_wc": "Variación circulante",
    }
    st.dataframe(datos.anual.rename(columns=nombres).T.style.format("{:,.2f}", na_rep="—"),
                 use_container_width=True)
    hist = rt.historico_multiplos(datos)
    if not hist.empty:
        st.markdown("**Múltiplos a cierre de cada año**")
        st.dataframe(hist.T.style.format("{:,.2f}", na_rep="—"), use_container_width=True)
