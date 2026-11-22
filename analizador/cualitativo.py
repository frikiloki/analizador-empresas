"""Análisis cualitativo: lista de comprobación de la tabla, datos de insiders
y, si hay clave de Anthropic, un análisis redactado por IA."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .datos import DatosEmpresa

# Bloques de la tabla de valoración (página "ANÁLISIS CUALITATIVO")
CHECKLIST: dict[str, list[str]] = {
    "1. Modelo de negocio": [
        "Ciclicidad", "Ramas del negocio", "Canales de venta", "Servicio / producto",
    ],
    "2. Entorno macro (PESTEL)": [
        "Políticos", "Económicos", "Sociales", "Tecnológicos", "Ecológicos", "Legales",
    ],
    "3. Entorno micro": [
        "Competidores: tamaño", "Competidores: crecimiento", "Competidores: ventajas competitivas",
        "Competidores: últimos 5 años", "Competidores: canales de distribución",
        "Competidores: equipos directivos", "Competidores: salud financiera",
        "Competidores: estrategia de precios", "Competidores: cultura de trabajo",
        "Competidores: gasto en I+D", "Complementadores", "Sustitutos", "Posibles entrantes",
        "Concentración de proveedores", "Concentración de clientes",
        "Acreedores y accionistas (nivel retail / instituciones)",
    ],
    "4. Equipo directivo": [
        "Visión a largo plazo", "Fundadores en la empresa", "Trayectoria y cambios recientes",
    ],
    "5. Moat": [
        "Cultura organizacional sólida (pertenencia, orientación al cliente, know-how)",
        "Adaptabilidad (innovación, adopción selectiva de tecnología, diversificación razonable)",
        "Reinversión en I+D y producto (capacidad, feedback de clientes, propiedad intelectual)",
        "Reputación (confianza, atracción de talento, blindaje ante escándalos)",
        "Visión a largo plazo y liderazgo (proyectos largos, no canibalizar marca, sucesión)",
        "Poder de precios", "Marca", "Poder de negociación con proveedores",
        "Coste de sustitución", "Efecto red", "Patentes", "Situación geográfica",
        "Bienes Veblen",
    ],
}


def resumen_insiders(datos: DatosEmpresa, meses: int = 12) -> dict:
    """Cuenta compras y ventas de insiders en los últimos `meses`."""
    t = datos.insiders
    out = {"compras": 0, "ventas": 0, "acc_compradas": 0.0, "acc_vendidas": 0.0,
           "importe_compras": 0.0, "importe_ventas": 0.0, "hay_datos": not t.empty}
    if t.empty or "Fecha" not in t:
        return out
    desde = pd.Timestamp.today() - pd.DateOffset(months=meses)
    t = t[t["Fecha"] >= desde]
    texto = (t.get("Operación", pd.Series("", index=t.index)).fillna("").astype(str) + " " +
             t.get("Detalle", pd.Series("", index=t.index)).fillna("").astype(str)).str.lower()
    compra = texto.str.contains("purchase|buy|compra")
    venta = texto.str.contains("sale|sell|venta") & ~compra
    acc = pd.to_numeric(t.get("Acciones", 0), errors="coerce").fillna(0)
    imp = pd.to_numeric(t.get("Importe", 0), errors="coerce").fillna(0)
    out.update(
        compras=int(compra.sum()), ventas=int(venta.sum()),
        acc_compradas=float(acc[compra].sum()), acc_vendidas=float(acc[venta].sum()),
        importe_compras=float(imp[compra].sum()), importe_ventas=float(imp[venta].sum()),
    )
    return out


def valoracion_skin_in_the_game(pct_insiders) -> tuple[str, str]:
    if pct_insiders is None or pd.isna(pct_insiders):
        return "info", "Sin datos del porcentaje en manos de insiders."
    p = float(pct_insiders)
    if p >= 0.10:
        return "ok", "Directivos y consejeros con mucho capital propio en juego."
    if p >= 0.03:
        return "ok", "Participación relevante de los insiders."
    if p >= 0.01:
        return "aviso", "Participación pequeña de los insiders."
    return "mal", "Los insiders apenas tienen acciones (habitual en empresas muy grandes)."


def ciclicidad(datos: DatosEmpresa) -> str:
    df = datos.anual
    s = df["revenue"].dropna()
    if len(s) < 4:
        return "Sin datos suficientes."
    var = s.pct_change().dropna()
    caidas = int((var < 0).sum())
    vol = float(var.std())
    beta = datos.perfil.get("beta")
    txt = f"Ventas: {caidas} años de caída de {len(var)}; volatilidad del crecimiento {vol * 100:.0f} %"
    if beta not in (None, "") and pd.notna(beta):
        txt += f"; beta {float(beta):.2f}"
    if caidas == 0 and vol < 0.10:
        txt += ". Perfil poco cíclico."
    elif caidas >= 2 or vol > 0.20:
        txt += ". Perfil cíclico."
    else:
        txt += ". Ciclicidad moderada."
    return txt


# ---------------------------------------------------------------------------
# Análisis con IA (opcional)
# ---------------------------------------------------------------------------

NOTICIAS = "Noticias recientes (últimos 12 meses)"

# Qué datos de la app necesita cada bloque (para no enviar de más y gastar menos)
_DATOS_POR_BLOQUE = {
    "1. Modelo de negocio": {"descripcion", "ratios"},
    "2. Entorno macro (PESTEL)": {"descripcion"},
    "3. Entorno micro": {"descripcion", "ratios", "accionistas"},
    "4. Equipo directivo": {"directivos", "insiders", "accionistas"},
    "5. Moat": {"descripcion", "ratios"},
}
# Bloques en los que buscar en internet aporta de verdad
_BLOQUES_CON_BUSQUEDA = {"2. Entorno macro (PESTEL)", "3. Entorno micro", "4. Equipo directivo", NOTICIAS}


def _contexto(datos: DatosEmpresa, resultados, notas: dict, seleccion: dict) -> str:
    necesita = set().union(*[_DATOS_POR_BLOQUE.get(b, set()) for b in seleccion]) if seleccion else set()
    p = datos.perfil
    ctx = {"empresa": datos.nombre, "ticker": datos.ticker, "sector": p.get("sector"),
           "industria": p.get("industria"), "pais": p.get("pais"), "empleados": p.get("empleados"),
           "ceo": p.get("ceo")}
    if "descripcion" in necesita and p.get("descripcion"):
        ctx["descripcion"] = str(p["descripcion"])[:1500]
    if "ratios" in necesita:
        ctx["ratios"] = [f"{r.nombre}: {r.valor} ({r.estado})" for r in resultados if r.estado != "info"]
    if "accionistas" in necesita:
        ctx["accionistas"] = datos.accionistas
    if "insiders" in necesita:
        ctx["insiders_12_meses"] = {k: v for k, v in resumen_insiders(datos).items() if k != "hay_datos"}
    if "directivos" in necesita and not datos.directivos.empty:
        cols = [c for c in ("Nombre", "Cargo", "Año nac.", "En el cargo desde") if c in datos.directivos]
        ctx["directivos"] = datos.directivos[cols].head(8).to_dict(orient="records")
    notas_sel = {k: v for k, v in notas.items() if v and k.split("|")[0] in seleccion}
    if notas_sel:
        ctx["notas_del_usuario"] = notas_sel
    return json.dumps(ctx, ensure_ascii=False, default=str)


_PRECIOS = {"haiku": (0.10, 0.50), "opus": (4.0, 20.0), "sonnet": (2.0, 10.0)}


def _precio_modelo(modelo: str) -> tuple[float, float]:
    return next((v for k, v in _PRECIOS.items() if k in modelo), _PRECIOS["sonnet"])


def _n_criterios(seleccion: dict) -> int:
    return sum(max(1, len(v)) for v in seleccion.values())


def busqueda_util(seleccion: dict) -> bool:
    return any(b in _BLOQUES_CON_BUSQUEDA for b in seleccion)


def estimar_coste(seleccion: dict, modelo: str, buscar: bool, max_busquedas: int) -> float:
    """Estimación previa (en dólares) para mostrar antes de lanzar el análisis."""
    if not seleccion:
        return 0.0
    p_in, p_out = _precio_modelo(modelo)
    n = _n_criterios(seleccion)
    tok_in = 1500 + 400 * len(seleccion)
    tok_out = 180 * n + 150
    coste = tok_in / 1e6 * p_in + tok_out / 1e6 * p_out
    if buscar and busqueda_util(seleccion):
        b = min(max_busquedas, 1 + len(seleccion))
        coste += b * 0.01 + b * 6000 / 1e6 * p_in  # búsqueda + texto de los resultados
    return coste


def analisis_ia(datos: DatosEmpresa, resultados, notas: dict, clave: str, seleccion: dict,
                modelo: str = "claude-sonnet-5-5", buscar: bool = True, max_busquedas: int = 3) -> dict:
    """Analiza solo los bloques/criterios elegidos.

    `seleccion` = {bloque: [criterios]}; una lista vacía significa «el bloque entero, en resumen».
    Devuelve {"texto", "fuentes": [(titulo, url)], "busquedas", "coste", "seleccion", "modelo", "fecha"}.
    """
    import anthropic

    hoy = pd.Timestamp.today().strftime("%d/%m/%Y")
    buscar = buscar and busqueda_util(seleccion)

    lineas = []
    for bloque, crit in seleccion.items():
        if bloque == NOTICIAS:
            lineas.append(f"- {NOTICIAS}: las 5 más relevantes, cada una con fecha y una línea de por qué importa.")
        elif crit:
            lineas.append(f"- {bloque}, SOLO estos criterios: {'; '.join(crit)}")
        else:
            lineas.append(f"- {bloque} (visión general del bloque en un párrafo)")

    sistema = (
        "Eres un analista de inversiones que ayuda a un inversor particular. Respondes en español, "
        "breve y al grano: frases cortas, sin introducciones ni repetir datos que ya te paso. "
        "No das recomendaciones de compra o venta. Si no sabes algo con seguridad, dilo y no inventes cifras. "
        f"Hoy es {hoy}. "
        + ("Usa la búsqueda web solo para hechos recientes que cambien la valoración; no busques lo que ya sabes."
           if buscar else "No tienes internet: avisa cuando un punto dependa de hechos recientes.")
    )
    pregunta = (
        "Analiza ÚNICAMENTE lo siguiente de esta empresa (no añadas otros bloques):\n"
        + "\n".join(lineas)
        + "\n\nFormato: un encabezado «### » por bloque; dentro, cada criterio en negrita con 2-3 frases y "
        "al final de cada criterio «→ Favorable», «→ Neutral» o «→ Desfavorable». "
        "En equipo directivo usa los datos de insiders que te paso (no comentes sueldos ni n.º de acciones: "
        "el usuario ya los ve en la app)."
        + f"\n\nDatos:\n{_contexto(datos, resultados, notas, seleccion)}"
    )

    cliente = anthropic.Anthropic(api_key=clave)
    mensajes = [{"role": "user", "content": pregunta}]
    herramientas = ([{"type": "web_search_20250305", "name": "web_search", "max_uses": max_busquedas}]
                    if buscar else [])
    max_tokens = min(6000, 400 + 350 * _n_criterios(seleccion))

    textos, fuentes = [], []
    busquedas, tok_in, tok_out = 0, 0, 0
    for _ in range(4):  # la búsqueda puede pausar el turno; se continúa hasta terminar
        kwargs = dict(model=modelo, max_tokens=max_tokens, system=sistema, messages=mensajes)
        if herramientas:
            kwargs["tools"] = herramientas
        resp = cliente.messages.create(**kwargs)
        u = resp.usage
        tok_in += getattr(u, "input_tokens", 0) or 0
        tok_out += getattr(u, "output_tokens", 0) or 0
        stu = getattr(u, "server_tool_use", None)
        busquedas += (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
        for b in resp.content:
            if getattr(b, "type", "") == "text":
                textos.append(b.text)
                for c in getattr(b, "citations", None) or []:
                    url = getattr(c, "url", None)
                    if url and url not in [f[1] for f in fuentes]:
                        fuentes.append((getattr(c, "title", None) or url, url))
        if resp.stop_reason != "pause_turn":
            break
        mensajes = mensajes + [{"role": "assistant", "content": resp.content}]

    p_in, p_out = _precio_modelo(modelo)
    coste = tok_in / 1e6 * p_in + tok_out / 1e6 * p_out + busquedas * 0.01
    return {"texto": "".join(textos), "fuentes": fuentes, "busquedas": busquedas, "coste": coste,
            "seleccion": seleccion, "modelo": modelo, "fecha": pd.Timestamp.today().strftime("%Y-%m-%d %H:%M")}


def exportar_markdown(datos: DatosEmpresa, resultados, notas: dict, valoraciones: dict, texto_ia: str = "") -> str:
    simbolo = {"ok": "✅", "aviso": "🟡", "mal": "❌", "info": "⚪"}
    lineas = [f"# {datos.nombre} ({datos.ticker})", "",
              f"Fecha: {pd.Timestamp.today():%d/%m/%Y} · Fuente: {datos.fuente}", "",
              "## Análisis cuantitativo", "", "| | Punto | Ratio | Valor | Ideal | Comentario |",
              "|---|---|---|---|---|---|"]
    for r in resultados:
        lineas.append(f"| {simbolo[r.estado]} | {r.punto} | {r.nombre} | {r.valor} | {r.ideal} | {r.comentario} |")
    lineas += ["", "## Análisis cualitativo", ""]
    for bloque, items in CHECKLIST.items():
        lineas.append(f"### {bloque}")
        for it in items:
            clave = f"{bloque}|{it}"
            v = valoraciones.get(clave, "—")
            n = notas.get(clave, "")
            lineas.append(f"- **{it}**: {v}" + (f" — {n}" if n else ""))
        lineas.append("")
    if texto_ia:
        lineas += ["## Análisis cualitativo con IA", "", texto_ia]
    return "\n".join(lineas)
