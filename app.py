"""Analizador de empresas según la tabla de valoración propia.

Ejecutar en local:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador import cualitativo as cual
from analizador.almacen import Almacen, crear_registro
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
MODELOS_IA = {"Sonnet (más completo)": MODELO_IA, "Haiku (≈20 veces más barato)": "claude-haiku-5-5"}
ALMACEN = Almacen(secreto("GITHUB_TOKEN"), secreto("GITHUB_REPO_DATOS"))
OPCIONES_CHECK = ["—", "✅ Favorable", "🟡 Neutral", "❌ Desfavorable"]


@st.cache_data(ttl=60, show_spinner=False)
def versiones_guardadas(tk: str) -> list[str]:
    try:
        return ALMACEN.versiones(tk)
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def tickers_guardados() -> list[str]:
    try:
        return ALMACEN.tickers()
    except Exception:
        return []


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def registro_guardado(tk: str, nombre: str) -> dict:
    return ALMACEN.cargar(tk, nombre)


def _k_valor(tk, k):
    return f"v_{tk}_{k}"


def _k_nota(tk, k):
    return f"n_{tk}_{k}"


def leer_checklist(tk: str) -> tuple[dict, dict]:
    notas, valoraciones = {}, {}
    for bloque, items in cual.CHECKLIST.items():
        for it in items:
            k = f"{bloque}|{it}"
            valoraciones[k] = st.session_state.get(_k_valor(tk, k), "—")
            notas[k] = st.session_state.get(_k_nota(tk, k), "")
    return notas, valoraciones


def poner_checklist(tk: str, notas: dict, valoraciones: dict) -> None:
    for bloque, items in cual.CHECKLIST.items():
        for it in items:
            k = f"{bloque}|{it}"
            v = valoraciones.get(k, "—")
            st.session_state[_k_valor(tk, k)] = v if v in OPCIONES_CHECK else "—"
            st.session_state[_k_nota(tk, k)] = notas.get(k, "")


def texto_ia_completo(lista: list) -> str:
    partes = []
    for ia in lista:
        t = f"*{ia.get('fecha', '')} · {ia.get('modelo', '')}*\n\n{ia.get('texto', '')}"
        if ia.get("fuentes"):
            t += "\n\n**Fuentes**\n" + "\n".join(f"- [{a}]({b})" for a, b in ia["fuentes"])
        partes.append(t)
    return "\n\n---\n\n".join(partes)


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
        + "  \n" + ("🟢 Guardado en GitHub" if ALMACEN.persistente else "🟡 Guardado temporal")
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

CLAVE_LISTA_IA = f"ia_{ticker}"
st.session_state.setdefault(CLAVE_LISTA_IA, [])

g1, g2 = st.columns([1, 3])
if g1.button("💾 Guardar análisis", use_container_width=True,
             help="Guarda ratios, checklist, notas y análisis de IA con la fecha de hoy para compararlo más adelante."):
    notas_g, val_g = leer_checklist(ticker)
    registro = crear_registro(datos, resultados, res, rt.tabla_comparativa(datos), notas_g, val_g,
                              st.session_state[CLAVE_LISTA_IA])
    try:
        ALMACEN.guardar(registro)
        versiones_guardadas.clear()
        tickers_guardados.clear()
        g2.success(f"Guardado ({registro['fecha']}). Lo tienes en la pestaña Historial.")
    except Exception as e:
        g2.error(f"No se ha podido guardar: {e}")
if not ALMACEN.persistente:
    g2.caption("⚠️ Sin repositorio de datos configurado: lo guardado se pierde cuando la app se reinicia.")

tab_cuanti, tab_graf, tab_cuali, tab_comp, tab_hist, tab_datos = st.tabs(
    ["Cuantitativo", "Gráficos", "Cualitativo", "Competidores", "Historial", "Datos"])

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
    lista_ia = st.session_state[CLAVE_LISTA_IA]

    # Recuperar lo guardado (antes de dibujar la checklist)
    versiones = versiones_guardadas(ticker)
    if versiones:
        r1, r2 = st.columns([2, 3])
        if r1.button("📂 Recuperar mi último análisis guardado", use_container_width=True,
                     help="Carga tus valoraciones, notas y análisis de IA de la última vez (sin volver a pagar la IA)."):
            try:
                reg = registro_guardado(ticker, versiones[0])
                poner_checklist(ticker, reg.get("notas", {}), reg.get("valoraciones", {}))
                ia_g = reg.get("ia") or []
                st.session_state[CLAVE_LISTA_IA] = [ia_g] if isinstance(ia_g, dict) and ia_g else list(ia_g)
                lista_ia = st.session_state[CLAVE_LISTA_IA]
                r2.success(f"Recuperado el análisis del {reg['fecha']}.")
            except Exception as e:
                r2.error(f"No se ha podido recuperar: {e}")
        else:
            r2.caption(f"Tienes {len(versiones)} análisis guardado(s) de esta empresa.")

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

    # ------------------------------------------------------------------ IA
    st.divider()
    st.subheader("Análisis con IA")
    if CLAVE_IA:
        st.caption("Elige solo lo que te interesa de esta empresa: pagas únicamente por lo que se analiza.")
        seleccion: dict[str, list[str]] = {}
        bloques_ia = list(cual.CHECKLIST.items()) + [(cual.NOTICIAS, [])]
        cols_sel = st.columns(2)
        for i, (bloque, items) in enumerate(bloques_ia):
            with cols_sel[i % 2]:
                if st.checkbox(bloque, key=f"ia_sel_{ticker}_{bloque}"):
                    if items:
                        crit = st.multiselect("Criterios (vacío = visión general del bloque)", items,
                                              key=f"ia_crit_{ticker}_{bloque}", label_visibility="collapsed",
                                              placeholder="Todos (resumen del bloque) o elige criterios")
                    else:
                        crit = []
                    seleccion[bloque] = crit

        o1, o2, o3 = st.columns([2, 2, 1])
        nombre_modelo = o1.radio("Modelo", list(MODELOS_IA), horizontal=True)
        modelo = MODELOS_IA[nombre_modelo]
        util = cual.busqueda_util(seleccion)
        buscar = o2.toggle("Buscar en internet", value=util, disabled=not util,
                           help="Solo tiene sentido para entorno macro y micro, equipo directivo y noticias. "
                                "Cada búsqueda cuesta 0,01 $ más el texto que devuelve.")
        max_b = o3.number_input("Máx. búsquedas", 1, 10, 3, disabled=not (buscar and util))

        if seleccion:
            est = cual.estimar_coste(seleccion, modelo, buscar, int(max_b))
            st.markdown(f"**Coste estimado: ≈ {est:.3f} $** ({est * 100:.1f} céntimos)")
        if st.button("Analizar con IA", type="primary", disabled=not seleccion) and seleccion:
            with st.spinner("Analizando… (puede tardar un minuto)"):
                resultado_ia = None
                try:
                    resultado_ia = cual.analisis_ia(datos, resultados, leer_checklist(ticker)[0], CLAVE_IA,
                                                    seleccion, modelo, buscar=buscar, max_busquedas=int(max_b))
                except Exception as e:
                    if buscar:
                        st.warning("La búsqueda web ha fallado (puede que no esté activada en tu cuenta de Anthropic). "
                                   f"Se repite sin búsqueda. Detalle: {e}")
                        try:
                            resultado_ia = cual.analisis_ia(datos, resultados, leer_checklist(ticker)[0], CLAVE_IA,
                                                            seleccion, modelo, buscar=False)
                        except Exception as e2:
                            st.error(f"No se ha podido generar el análisis: {e2}")
                    else:
                        st.error(f"No se ha podido generar el análisis: {e}")
                if resultado_ia and resultado_ia.get("texto"):
                    lista_ia.insert(0, resultado_ia)

        if lista_ia:
            total = sum(x.get("coste", 0) for x in lista_ia)
            st.caption(f"{len(lista_ia)} análisis de IA para esta empresa · gasto acumulado ≈ {total:.3f} $. "
                       "Pulsa «💾 Guardar análisis» arriba para no perderlos.")
        for i, ia in enumerate(list(lista_ia)):
            titulo = " · ".join(ia.get("seleccion", {}).keys()) or "Análisis"
            with st.expander(f"{ia.get('fecha', '')} — {titulo}", expanded=(i == 0)):
                st.markdown(ia.get("texto", ""))
                if ia.get("fuentes"):
                    st.markdown("**Fuentes**")
                    for t_, u_ in ia["fuentes"]:
                        st.markdown(f"- [{t_}]({u_})")
                st.caption(f"{ia.get('modelo', '')} · {ia.get('busquedas', 0)} búsquedas · "
                           f"coste ≈ {ia.get('coste', 0):.3f} $")
                if st.button("🗑️ Quitar", key=f"del_ia_{ticker}_{i}_{ia.get('fecha', '')}"):
                    lista_ia.remove(ia)
                    st.rerun()
    else:
        st.caption("Opcional: añade `ANTHROPIC_API_KEY` en los secretos de Streamlit para que la IA analice "
                   "los bloques que elijas de tu tabla.")

    # ------------------------------------------------------------------ Checklist
    st.divider()
    st.subheader("Tu checklist")
    st.caption("Valora cada punto y añade notas. Pulsa «💾 Guardar análisis» arriba para conservarlo.")
    for bloque, items in cual.CHECKLIST.items():
        with st.expander(bloque):
            for it in items:
                k = f"{bloque}|{it}"
                kv, kn = _k_valor(ticker, k), _k_nota(ticker, k)
                st.session_state.setdefault(kv, "—")
                st.session_state.setdefault(kn, "")
                c1, c2 = st.columns([2, 3])
                c1.selectbox(it, OPCIONES_CHECK, key=kv)
                c2.text_input("Notas", key=kn, label_visibility="hidden")

    notas_e, val_e = leer_checklist(ticker)
    informe = cual.exportar_markdown(datos, resultados, notas_e, val_e, texto_ia_completo(lista_ia))
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
# Historial
# ---------------------------------------------------------------------------
RANGO = {"mal": 0, "aviso": 1, "ok": 2, "info": None}

with tab_hist:
    st.caption(ALMACEN.comprobar() if ALMACEN.persistente else
               "⚠️ Guardado temporal: configura GITHUB_TOKEN y GITHUB_REPO_DATOS para conservar el historial.")
    todos = tickers_guardados()
    if not todos:
        st.info("Aún no has guardado ningún análisis. Pulsa «💾 Guardar análisis» arriba.")
    else:
        tk_h = st.selectbox("Empresa", todos, index=todos.index(ticker) if ticker in todos else 0)
        vers = versiones_guardadas(tk_h)[:15]
        regs = []
        for v in vers:
            try:
                regs.append(registro_guardado(tk_h, v))
            except Exception:
                pass
        if not regs:
            st.warning("No se han podido leer los análisis guardados.")
        else:
            # Evolución
            st.markdown("**Evolución**")
            filas = []
            for r_ in regs:
                m_ = r_.get("metricas", {})
                rs = r_.get("resumen", {})
                filas.append({
                    "Fecha": r_["fecha"], "Precio": r_.get("precio"),
                    "Puntuación": (rs.get("puntuacion") or 0) * 100 if rs.get("puntuacion") is not None else None,
                    "✅": rs.get("ok"), "🟡": rs.get("aviso"), "❌": rs.get("mal"),
                    "PER": m_.get("PER"), "EV/EBITDA": m_.get("EV/EBITDA"), "ROIC": m_.get("ROIC"),
                    "Margen neto": m_.get("Margen neto"), "Deuda neta/EBITDA": m_.get("Deuda neta/EBITDA"),
                    "IA": len(r_.get("ia") or []) if isinstance(r_.get("ia"), list) else int(bool(r_.get("ia"))),
                })
            evo = pd.DataFrame(filas).set_index("Fecha")
            st.dataframe(evo.style.format({"Precio": "{:,.2f}", "Puntuación": "{:.0f}", "PER": "{:.1f}x",
                                           "EV/EBITDA": "{:.1f}x", "ROIC": "{:.1%}", "Margen neto": "{:.1%}",
                                           "Deuda neta/EBITDA": "{:.2f}"}, na_rep="—"),
                         use_container_width=True)

            # Comparar dos fechas
            fechas_h = [r_["fecha"] for r_ in regs]
            if len(regs) >= 2:
                st.markdown("**Comparar dos análisis**")
                x1, x2 = st.columns(2)
                fa = x1.selectbox("Antes", fechas_h, index=len(fechas_h) - 1)
                fb = x2.selectbox("Después", fechas_h, index=0)
                ra_ = next(r_ for r_ in regs if r_["fecha"] == fa)
                rb_ = next(r_ for r_ in regs if r_["fecha"] == fb)
                mapa_a = {(x["punto"], x["nombre"]): x for x in ra_.get("resultados", [])}
                comp_f = []
                for x in rb_.get("resultados", []):
                    y = mapa_a.get((x["punto"], x["nombre"]))
                    ea, eb = (RANGO.get(y["estado"]) if y else None), RANGO.get(x["estado"])
                    if ea is None or eb is None:
                        cambio = ""
                    else:
                        cambio = "⬆️ mejora" if eb > ea else ("⬇️ empeora" if eb < ea else "=")
                    comp_f.append({"N.º": x["punto"], "Ratio": x["nombre"],
                                   f"Antes ({fa[:10]})": (ICONOS[y["estado"]] + " " + y["valor"]) if y else "—",
                                   f"Después ({fb[:10]})": ICONOS[x["estado"]] + " " + x["valor"],
                                   "Cambio": cambio})
                comp_df = pd.DataFrame(comp_f)
                solo_cambios = st.toggle("Ver solo lo que ha cambiado de estado", value=False)
                if solo_cambios:
                    comp_df = comp_df[comp_df["Cambio"].isin(["⬆️ mejora", "⬇️ empeora"])]
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Ver un análisis concreto
            st.markdown("**Ver un análisis guardado**")
            fv = st.selectbox("Fecha", fechas_h, key="hist_ver")
            rv = next(r_ for r_ in regs if r_["fecha"] == fv)
            vals = {k: v for k, v in (rv.get("valoraciones") or {}).items()}
            notas_v = rv.get("notas") or {}
            if vals or notas_v:
                with st.expander("Checklist y notas"):
                    for k in sorted(set(vals) | set(notas_v)):
                        bloque, it = k.split("|", 1)
                        st.markdown(f"- **{it}** ({bloque}): {vals.get(k, '—')}"
                                    + (f" — {notas_v[k]}" if notas_v.get(k) else ""))
            ia_v = rv.get("ia") or []
            ia_v = [ia_v] if isinstance(ia_v, dict) else ia_v
            for ia in ia_v:
                with st.expander(f"IA · {ia.get('fecha', '')} — {' · '.join(ia.get('seleccion', {}).keys())}"):
                    st.markdown(ia.get("texto", ""))
                    for t_, u_ in ia.get("fuentes") or []:
                        st.markdown(f"- [{t_}]({u_})")

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
