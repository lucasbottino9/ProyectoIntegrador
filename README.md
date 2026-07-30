# ProyectoIntegrador
# Proyecto Integrador MLOps - Sistema de Predicción de Pago de Créditos

# 1. Descripción del proyecto
Este proyecto consiste en la construcción de un pipeline completo de Machine Learning aplicado a un problema de clasificación binaria dentro del sector financiero.
El objetivo principal es desarrollar un sistema capaz de predecir si un cliente realizará el pago de un crédito a tiempo utilizando información histórica relacionada con:
- características del préstamo;
- perfil financiero del cliente;
- comportamiento crediticio;
- ingresos;
- niveles de endeudamiento;
- historial de mora.
El proyecto implementa buenas prácticas de MLOps incluyendo:
- organización modular del código;
- control de versiones con Git;
- separación de etapas del pipeline;
- ingeniería de características;
- entrenamiento y evaluación de modelos;
- monitoreo del modelo;
- detección de Data Drift;
- preparación para despliegue mediante aplicaciones interactivas.

# 2. Caso de negocio
## Contexto
Las entidades financieras necesitan evaluar continuamente el riesgo asociado a sus clientes antes y después de otorgar créditos
Uno de los principales desafíos consiste en identificar clientes con mayor probabilidad de incumplimiento para:
- reducir pérdidas económicas;
- mejorar la toma de decisiones;
- optimizar procesos de aprobación crediticia;
- realizar acciones preventivas sobre clientes con riesgo elevado.
Actualmente, muchas decisiones pueden depender de reglas estáticas que no capturan correctamente los cambios en el comportamiento de los clientes.
Por este motivo se propone desarrollar un modelo predictivo que permita estimar la probabilidad de pago oportuno de un crédito

# 3. Objetivo del modelo
El objetivo es construir un modelo de clasificación binaria que prediga

## Variable objetivo
Pago_atiempo
Donde:
- `1` representa que el cliente realizó el pago a tiempo.
- `0` representa incumplimiento o retraso en el pago.
- 
# 4. Dataset
El dataset contiene información histórica de clientes y créditos.
Cantidad de variables iniciales:
- Variables financieras.
- Variables laborales.
- Variables relacionadas con ingresos.
- Variables de comportamiento crediticio.
- Variables de historial de mora.
Principales variables:

| Variable | Descripción |
|---|---|
| tipo_credito | Tipo de crédito solicitado |
| fecha_prestamo | Fecha de otorgamiento |
| capital_prestado | Monto inicial del préstamo |
| plazo_meses | Duración del crédito |
| edad_cliente | Edad del cliente |
| tipo_laboral | Tipo de empleo |
| salario_cliente | Ingreso mensual |
| saldo_mora | Valor pendiente en mora |
| saldo_total | Deuda total |
| puntaje | Score interno |
| puntaje_datacredito | Score externo |
| Pago_atiempo | Variable objetivo |

# 5. Análisis exploratorio (EDA)
Durante la etapa inicial se realizó un análisis exploratorio completo para comprender la estructura del dataset.

## Exploración inicial
Se analizaron:
- dimensiones del dataset;
- tipos de variables;
- estadísticas descriptivas;
- distribución de valores.
Funciones utilizadas:
- `head()`
- `info()`
- `describe()`

# 6. Limpieza y preparación de datos
Se realizaron procesos de:
- detección de valores nulos;
- búsqueda de registros duplicados;
- conversión de tipos de datos;
- transformación de fechas;
- revisión de valores inconsistentes.
La fecha del préstamo fue transformada para obtener nuevas variables temporales.
Ejemplo:
dias_desde_prestamo

# 7. Ingeniería de características
Para mejorar la capacidad predictiva del modelo se generaron nuevas variables derivadas.
Principales features creadas:

## Capacidad de pago
ratio_cuota_salario
Representa qué proporción del ingreso mensual está comprometida con la cuota.

## Nivel de endeudamiento
ratio_deuda_salario
Permite medir la carga financiera del cliente.

## Deuda total
mora_total
Combina diferentes fuentes de deuda pendiente.

## Historial crediticio
total_creditos
Cantidad total de créditos registrados.

## Comparación de ingresos
ratio_salario_datacredito
Compara el salario declarado contra ingresos estimados por Datacrédito.

## Diferencia de puntajes
diferencia_puntaje
Mide discrepancias entre diferentes fuentes de scoring.

# 8. Preprocesamiento
El procesamiento de datos fue realizado utilizando herramientas de Scikit-Learn.

## Variables numéricas
Tratamiento:
- imputación de valores faltantes;
- escalamiento mediante
StandardScaler


## Variables categóricas

Variables transformadas:
- tipo_laboral;
- tendencia_ingresos.
Método utilizado:
OneHotEncoder

## Arquitectura utilizada
Se implementó:
ColumnTransformer y Pipeline
permitiendo mantener un flujo reproducible y preparado para producción.

# 9. Modelos evaluados
Se entrenarán diferentes algoritmos de clasificación:

## Regresión logística
Modelo base para clasificación binaria.

## Árbol de decisión
Permite interpretar reglas de decisión.

## Random Forest
Modelo basado en múltiples árboles con mejor capacidad de generalización.

## Gradient Boosting
Modelo ensemble orientado a mejorar precisión predictiva.

# 10. Evaluación del modelo
Los modelos serán comparados utilizando:

## Métricas principales

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

También se generan:
- matriz de confusión;
- curva ROC;
- reporte de clasificación;
- tabla comparativa de resultados.

# 11. Monitoreo del modelo
Una vez desplegado un modelo de Machine Learning es necesario controlar que los datos nuevos mantengan características similares a los datos utilizados durante entrenamiento.
Para esto se implementó:
model_monitoring.py

# 12. Detección de Data Drift
El sistema analiza cambios en la distribución de variables mediante:

## Kolmogorov-Smirnov Test
Utilizado para variables numéricas.
Permite detectar diferencias estadísticamente significativas entre dos distribuciones.

## Population Stability Index (PSI)
Clasificación utilizada:
| PSI | Interpretación |
|-|-|
| < 0.10 | Sin drift significativo |
| 0.10 - 0.25 | Drift moderado |
| > 0.25 | Drift crítico |

## Jensen-Shannon Divergence
Mide la diferencia entre distribuciones probabilísticas.

## Chi-Square Test
Aplicado sobre variables categóricas.
Permite identificar cambios en la frecuencia de categorías.

# 13. Visualización del monitoreo
El sistema genera:
- gráficos comparativos histórico vs actual;
- tablas de métricas;
- indicadores de riesgo;
- alertas de drift.
Esto permite identificar:
- cambios abruptos en la población;
- deterioro potencial del modelo;
- necesidad de reentrenamiento.
# 14. Flujo MLOps implementado
Datos históricos

        |
        v

Análisis exploratorio

        |
        v

Feature Engineering

        |
        v

Preprocesamiento

        |
        v

Entrenamiento modelos

        |
        v

Evaluación

        |
        v

Despliegue

        |
        v

Monitoreo Data Drift

        |
        v

Retraining

# 15. Próximos pasos
Como evolución del proyecto se contempla
- desarrollo de aplicación Streamlit;
- integración del modelo entrenado;
- creación de endpoints de predicción;
- almacenamiento histórico de métricas;
- automatización del monitoreo;
- estrategia automática de reentrenamiento.

# 16. Conclusión
El proyecto implementa un flujo completo de Machine Learning siguiendo principios de MLOps.
La solución permite pasar desde datos históricos hasta un sistema preparado para producción, incorporando:
- calidad de código;
- reproducibilidad;
- evaluación objetiva;
- monitoreo continuo;
- detección temprana de cambios en los datos
Esto permite mantener modelos confiables y adaptados a la evolución del comportamiento financiero de los clientes.
