"""Guardado de análisis para consultarlos y compararlos más adelante.

Si hay secretos GITHUB_TOKEN y GITHUB_REPO_DATOS (p. ej. "usuario/analizador-datos"),
cada análisis se guarda como un archivo JSON en ese repositorio (recomendado: privado):

    analisis/<TICKER>/<AAAA-MM-DD_HHMM>.json

Si no, se guarda en la carpeta local `datos_guardados/`, que en Streamlit Cloud
se borra cuando la app se reinicia (sirve solo para probar).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

API = "https://api.github.com"
CARPETA_LOCAL = Path("datos_guardados")


class Almacen:
    def __init__(self, token: str | None = None, repo: str | None = None):
        self.token = token
        self.repo = repo.strip().strip("/") if repo else None

    @property
    def persistente(self) -> bool:
        return bool(self.token and self.repo)

    # ------------------------------------------------------------------ GitHub
    def _h(self):
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def _gh_get(self, ruta: str):
        r = requests.get(f"{API}/repos/{self.repo}/contents/{ruta}", headers=self._h(), timeout=20)
        if r.status_code == 404:
            return None
        if r.status_code in (401, 403):
            raise PermissionError("GitHub ha rechazado el acceso: revisa el token y sus permisos (Contents: Read and write).")
        r.raise_for_status()
        return r.json()

    def comprobar(self) -> str:
        """Devuelve un texto con el estado de la conexión."""
        if not self.persistente:
            return "Guardado temporal (se borra al reiniciar la app)."
        r = requests.get(f"{API}/repos/{self.repo}", headers=self._h(), timeout=20)
        if r.status_code == 200:
            return f"Guardando en GitHub: {self.repo}" + (" (privado)" if r.json().get("private") else " (¡público!)")
        return f"No se puede acceder a {self.repo} (error {r.status_code}). Revisa el token y el nombre."

    # ------------------------------------------------------------------ API pública
    def guardar(self, registro: dict) -> str:
        ticker = registro["ticker"].replace("/", "_")
        nombre = registro["fecha"].replace(":", "").replace(" ", "_")
        ruta = f"analisis/{ticker}/{nombre}.json"
        contenido = json.dumps(registro, ensure_ascii=False, indent=1, default=str)
        if self.persistente:
            existente = self._gh_get(ruta)
            cuerpo = {"message": f"Análisis {ticker} {registro['fecha']}",
                      "content": base64.b64encode(contenido.encode("utf-8")).decode()}
            if existente and isinstance(existente, dict):
                cuerpo["sha"] = existente["sha"]
            r = requests.put(f"{API}/repos/{self.repo}/contents/{ruta}", headers=self._h(),
                             json=cuerpo, timeout=30)
            if r.status_code in (401, 403, 404):
                raise PermissionError("No se ha podido guardar en GitHub: revisa que el token tenga permiso "
                                      "de escritura (Contents: Read and write) en el repositorio de datos.")
            r.raise_for_status()
        else:
            p = CARPETA_LOCAL / ruta
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(contenido, encoding="utf-8")
        return ruta

    def tickers(self) -> list[str]:
        if self.persistente:
            datos = self._gh_get("analisis") or []
            return sorted(d["name"] for d in datos if d.get("type") == "dir")
        base = CARPETA_LOCAL / "analisis"
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []

    def versiones(self, ticker: str) -> list[str]:
        """Nombres de archivo de un ticker, del más reciente al más antiguo."""
        if self.persistente:
            datos = self._gh_get(f"analisis/{ticker}") or []
            nombres = [d["name"] for d in datos if d["name"].endswith(".json")]
        else:
            base = CARPETA_LOCAL / "analisis" / ticker
            nombres = [p.name for p in base.glob("*.json")] if base.exists() else []
        return sorted(nombres, reverse=True)

    def cargar(self, ticker: str, nombre: str) -> dict:
        ruta = f"analisis/{ticker}/{nombre}"
        if self.persistente:
            d = self._gh_get(ruta)
            if d is None:
                raise FileNotFoundError(ruta)
            if d.get("content"):
                texto = base64.b64decode(d["content"]).decode("utf-8")
            else:  # archivos grandes: descarga directa
                texto = requests.get(d["download_url"], headers=self._h(), timeout=30).text
            return json.loads(texto)
        return json.loads((CARPETA_LOCAL / ruta).read_text(encoding="utf-8"))


def crear_registro(datos, resultados, resumen: dict, metricas: dict, notas: dict,
                   valoraciones: dict, ia: dict) -> dict:
    import pandas as pd

    return {
        "fecha": pd.Timestamp.now(tz="Europe/Madrid").strftime("%Y-%m-%d %H:%M"),
        "ticker": datos.ticker,
        "nombre": datos.nombre,
        "fuente": datos.fuente,
        "divisa": datos.perfil.get("divisa"),
        "precio": datos.precio,
        "market_cap": datos.market_cap,
        "resumen": {k: (None if v != v else v) for k, v in resumen.items()},  # NaN -> None
        "metricas": {k: (None if isinstance(v, float) and v != v else v) for k, v in metricas.items()},
        "resultados": [r.__dict__ for r in resultados],
        "notas": {k: v for k, v in notas.items() if v},
        "valoraciones": {k: v for k, v in valoraciones.items() if v and v != "—"},
        "ia": ia or [],
    }
