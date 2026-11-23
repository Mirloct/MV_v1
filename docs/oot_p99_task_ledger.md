# Ledger de implementación: cola OOT P99

Última actualización: 2026-11-22

Objetivo: reducir el exportable y el dashboard operativo a individuos con score mayor o igual al percentil 99 de cada detector sobre el bloque OOT completo, y mejorar la lectura del análisis de cortes del Isolation Forest.

| ID | Tarea | Estado | Evidencia / criterio de cierre |
|---|---|---|---|
| T1 | Trazar configuración, CLI, exportable y dashboard | Completada | Excel P90+, dashboard/KPI/recurrencia P99 y checkpoint P95 diferenciados |
| T2 | Mantener Excel de control en P90+ | Completada | Default de API/configuración = 90; archivo `oot_p90_*` con bandas |
| T3 | Alinear dashboard operativo oficial con P99 | Completada | Universo, pestañas, recurrencia, etiquetas y pruebas usan P99 |
| T4 | Agregar heurística para empate entre mejores/peores variables | Completada | Interpretación neutral explícita en HTML y Markdown, con prueba de empate |
| T5 | Mejorar indicadores accionables del toolkit | Completada | Cobertura, rango de cortes, separación y recomendación de ablación |
| T6 | Depurar pruebas y documentación desactualizadas | Completada | 387 pruebas raíz + 70 enfocadas; smoke completo aprobado |
| T7 | Generar informe técnico HTML con diagrama Mermaid | Completada | `docs/oot_p99_technical_report.html` |
| T8 | Validación independiente, commit y push | Completada | Revisión independiente corregida; commit publicado en `origin/main` |

Punto seguro de reanudación: continuar desde la primera tarea no marcada como “Completada”; no modificar ni revertir cambios ajenos al alcance anterior.
