# Validación: capítulos "Diagnóstico cruzado IF-VAE" e "Interpretación"

Esta carpeta es evidencia de validación, generada el 2026-09-07, de que el
reporte `anomaly_report.html` producido por `main.py` (commit `2c13c0f` de
`main`) sí incluye los dos capítulos nuevos: la ficha estructurada
"Diagnóstico cruzado IF-VAE" y el capítulo "Interpretación y
recomendaciones" (toolkit por análisis, validación de indicadores, flujo
de decisión, recomendación al analista).

## Qué se validó (5 pasadas)

1. **Corrida fresca**: se borró el `anomaly_report.html` anterior y se
   corrió `python main.py --quick --no-tune` desde cero (sin ninguna
   bandera de diagnóstico -- el capítulo está encendido por defecto).
2. **Revisión del log**: `Phase 9c: IF-VAE diagnostic suite` corre sin
   advertencias ni errores, produce los cuadrantes reales
   (`BOTH=35, IF_ONLY=65, VAE_ONLY=73, NEITHER=1327`).
3. **Verificación de texto**: se confirmó que el HTML resultante contiene
   los `id` de ambos capítulos, los enlaces de navegación, y los
   encabezados de cada subsección (toolkit, validación de indicadores,
   flujo de decisión, recomendación).
4. **Verificación visual con navegador real** (Playwright/Chromium
   headless): se cargó el archivo, se hizo clic en los enlaces de
   navegación "Diagnóstico cruzado" e "Interpretación", y se confirmó
   mediante el DOM que ambas secciones son visibles (no ocultas por CSS) y
   sin errores de JavaScript en consola.
5. **Capturas de pantalla** (carpeta `capturas/`): evidencia visual directa
   de que el contenido real aparece en el navegador, no solo en el código
   fuente del HTML.

## Capturas

- `01_nav.png` -- barra de navegación con los enlaces "Diagnóstico cruzado"
  e "Interpretación".
- `02_diagnostic_suite_top.png` / `03_diagnostic_suite_body.png` /
  `04_diagnostic_suite_more.png` -- ficha estructurada: alcance,
  configuración efectiva, con datos reales de esta corrida.
- `05_interpretation_top.png` / `06_interpretation_toolkit.png` -- toolkit
  de interpretación por análisis, con severidad (Informativo/Atención) y
  base de cada afirmación.
- `07_interpretation_validation.png` -- validación de indicadores y el
  **flujo de decisión** con sus 4 nodos reales, coloreados por severidad,
  conectados por flechas.
- `08_interpretation_decisionflow.png` / `09_interpretation_recommendation.png`
  -- cierre del flujo y arranque de la sección "Modelos" del reporte.

## Si tu copia local no muestra estos capítulos

El motivo más probable es que estás abriendo un `anomaly_report.html`
generado **antes** de este commit. Bórralo y vuelve a correr:

```
python main.py --quick --no-tune
```

Si el capítulo sigue sin aparecer, revisa `artifacts/logs/execution.log`
buscando la línea `Phase 9c: IF-VAE diagnostic suite` -- si en cambio ves
`IF-VAE diagnostic suite failed (...)`, significa que el paquete vendorizado
no está instalado en ese entorno:

```
pip install -e tools/if_vae_diagnostic_suite
```

## Esta carpeta no es una versión oficial del proyecto

Es evidencia de validación de un punto en el tiempo. El reporte real se
regenera siempre corriendo el pipeline; no se versiona en `main` (vive en
`artifacts/`, que está en `.gitignore`).
