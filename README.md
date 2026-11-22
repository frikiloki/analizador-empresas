# Analizador de empresas

App de Streamlit que revisa una empresa a partir de su ticker con mi tabla de valoración:

- **Análisis cuantitativo**: los 20 puntos de la tabla, cada uno marcado como ✅ cumple, 🟡 a vigilar o ❌ no cumple.
- **Gráficos**: precio frente a BPA, magnitudes principales, márgenes, ROIC, acciones diluidas y múltiplos históricos.
- **Análisis cualitativo**: modelo de negocio, ciclicidad, I+D, *skin in the game*, compras y ventas de insiders, accionistas institucionales y la checklist completa (macro, micro, equipo directivo y moat), con informe descargable.
- **Competidores**: comparativa de márgenes, rentabilidad, deuda y múltiplos.
- **Historial**: guarda cada análisis con su fecha y compara cómo ha cambiado cada ratio entre dos fechas.

## Fuentes de datos

- **Financial Modeling Prep** (si hay clave): estados financieros, precios, perfil y directivos.
- **Yahoo Finance** (siempre): respaldo cuando FMP no cubre el ticker (por ejemplo, empresas europeas en el plan gratuito), y datos de insiders y accionistas.

Para acciones fuera de EE. UU. usa el sufijo de Yahoo: `ITX.MC`, `MC.PA`, `SAP.DE`…

## Secretos (Streamlit Cloud → Settings → Secrets)

```toml
FMP_API_KEY = "tu_clave"
# Opcional, para el análisis cualitativo con IA (por bloques o criterios):
ANTHROPIC_API_KEY = "tu_clave"
# Opcional, para guardar el historial de análisis en un repositorio PRIVADO:
GITHUB_TOKEN = "github_pat_..."
GITHUB_REPO_DATOS = "usuario/analizador-datos"
```

Nunca subas las claves al repositorio.

## Notas sobre los cálculos

- Punto 8: la tabla dice «Net debt / interest expenses», pero por los umbrales (< 0, < 2, > 3) se calcula **deuda neta / EBITDA**.
- Punto 16: el alfa es la rentabilidad anual de la acción menos la del índice elegido en 5 años (solo precio).
- Punto 19: «deuda total» se interpreta como **pasivo total**.
- Punto 20: «activo fijo» = activo no corriente.
- WACC: CAPM con la beta de la empresa; la tasa libre de riesgo y la prima de mercado se ajustan en la barra lateral.

Herramienta de apoyo al análisis. No es una recomendación de inversión.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
