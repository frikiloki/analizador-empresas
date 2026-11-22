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
        "Visión a largo plazo", "Fundadores en la empresa", "N.º de acciones de los directivos",
        "Sueldo de los directivos",
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

def _contexto(datos: DatosEmpresa, resultados, notas: dict) -> str:
    perfil = {k: v for k, v in datos.perfil.items() if k != "descripcion"}
    ins = resumen_insiders(datos)
    ratios = [f"{r.punto}. {r.nombre}: {r.valor} (ideal {r.ideal}) -> {r.estado}" for r in resultados]
    directivos = datos.directivos.head(10).to_dict(orient="records") if not datos.directivos.empty else []
    return json.dumps({
        "empresa": datos.nombre, "ticker": datos.ticker, "perfil": perfil,
        "descripcion": datos.perfil.get("descripcion"),
        "accionistas": datos.accionistas, "insiders_12_meses": ins,
        "directivos": directivos, "ratios": ratios,
        "notas_del_usuario": {k: v for k, v in notas.items() if v},
    }, ensure_ascii=False, default=str)


def analisis_ia(datos: DatosEmpresa, resultados, notas: dict, clave: str,
                modelo: str = "claude-sonnet-5-5", buscar: bool = True,
                max_busquedas: int = 5) -> dict:
    """Devuelve {"texto", "fuentes": [(titulo, url)], "busquedas", "coste"}."""
    import anthropic

    hoy = pd.Timestamp.today().strftime("%d/%m/%Y")
    bloques = "\n".join(f"- {b}: {', '.join(items)}" for b, items in CHECKLIST.items())
    sistema = (
        "Eres un analista de inversiones que ayuda a un inversor particular a revisar empresas. "
        "Respondes en español, de forma clara y concreta. No das recomendaciones de compra o venta: "
        "señalas fortalezas, riesgos y dudas a investigar. Si no sabes algo con seguridad, dilo "
        "expresamente y no inventes cifras. "
        f"Hoy es {hoy}. "
        + ("Usa la búsqueda web para comprobar hechos recientes y cita de dónde sale cada dato reciente."
           if buscar else
           "No tienes acceso a internet: tu conocimiento puede no estar actualizado, avísalo cuando un "
           "punto dependa de hechos recientes.")
    )
    tareas_busqueda = (
        "\n\nAntes de redactar, busca en internet (prioriza los últimos 12 meses):\n"
        "1. Noticias recientes relevantes de la empresa (resultados, guías, adquisiciones, demandas, "
        "regulación, escándalos).\n"
        "2. Cambios en el equipo directivo o el consejo, y su historial.\n"
        "3. Compras o ventas significativas de insiders y si forman parte de planes programados.\n"
        "4. Situación de los principales competidores y nuevos entrantes.\n"
        "5. Opinión del mercado sobre su ventaja competitiva.\n"
        "Termina con una sección «Noticias recientes» con las 5 más relevantes, cada una con su fecha."
        if buscar else ""
    )
    pregunta = (
        "Haz el análisis cualitativo de esta empresa siguiendo exactamente estos bloques de mi tabla:\n"
        f"{bloques}\n\n"
        "Para cada bloque: un párrafo breve y, al final, una valoración Favorable / Neutral / Desfavorable. "
        "En 'Equipo directivo' comenta el skin in the game y las compras y ventas de insiders con los datos "
        "que te paso. En 'Moat' indica qué ventajas competitivas tiene realmente y cuáles no. Termina con "
        "una lista de 3-5 preguntas que debería investigar antes de invertir."
        f"{tareas_busqueda}\n\n"
        f"Datos:\n{_contexto(datos, resultados, notas)}"
    )

    cliente = anthropic.Anthropic(api_key=clave)
    mensajes = [{"role": "user", "content": pregunta}]
    herramientas = []
    if buscar:
        herramientas = [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_busquedas}]

    textos, fuentes = [], []
    busquedas, tok_in, tok_out = 0, 0, 0
    for _ in range(4):  # la búsqueda puede pausar el turno; se continúa hasta terminar
        kwargs = dict(model=modelo, max_tokens=6000, system=sistema, messages=mensajes)
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

    # Coste orientativo con precios de Sonnet 5.5 (2 $ / 10 $ por millón) y 10 $ / 1.000 búsquedas
    precios = {"haiku": (0.10, 0.50), "opus": (4.0, 20.0)}
    p_in, p_out = next((v for k, v in precios.items() if k in modelo), (2.0, 10.0))
    coste = tok_in / 1e6 * p_in + tok_out / 1e6 * p_out + busquedas * 0.01
    return {"texto": "".join(textos), "fuentes": fuentes, "busquedas": busquedas, "coste": coste}


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
