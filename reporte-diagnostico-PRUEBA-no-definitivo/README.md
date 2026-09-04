# ⚠️ Reporte de prueba — NO es una versión definitiva

Este archivo (`anomaly_report.html`) es una **captura de prueba**, publicada
únicamente para mostrar la nueva estructura del capítulo **"Diagnóstico
cruzado IF-VAE"** del reporte principal. No reemplaza ni modifica ningún
reporte, dato o resultado existente en el repositorio.

## Qué es

- Generado con `python main.py --quick --no-tune --no-stack-iforest-into-vae
  --run-diagnostic-suite --diagnostic-entity-view` sobre datos **sintéticos**
  de desarrollo (500 individuos, corrida rápida de humo), no sobre datos
  oficiales.
- El único cambio relevante frente al reporte habitual es el capítulo
  "Diagnóstico cruzado IF-VAE": ahora es una ficha estructurada de 15
  secciones (alcance, configuración efectiva, matriz de disponibilidad,
  concordancia, candidatos VAE, sensibilidad, latente, autopsias, drift,
  estabilidad, temporal/segmentación, bloques condicionados a verdad base,
  experimentos, y procedencia), sin interpretaciones ni veredictos sobre qué
  detector es mejor.

## Qué NO es

- No es el reporte de una corrida oficial.
- No sustituye ningún artefacto de `artifacts/reports/` de este proyecto.
- No incorpora el resto de cambios de código de esta sesión (esos siguen sin
  subir, pendientes de evaluación por separado).

## Rama

Esta carpeta vive en la rama `reporte-diagnostico-prueba`, creada a partir de
`main` sin ningún otro commit pendiente. Se puede borrar cuando ya no haga
falta, sin afectar `main`.
