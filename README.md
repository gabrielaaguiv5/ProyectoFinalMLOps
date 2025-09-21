# ProyectoFinalMLOps

## Integrantes
- Gabriela Aguilar  
- Ximena Patiño Henao  
- Julián Ruiz  

---

## Descripción General
Este proyecto lo organizamos en varias carpetas dentro de `src/`, cada una con un rol muy claro en el ciclo de vida del modelo.  
La idea fue estructurarlo de manera que todo el flujo, desde que recibimos los datos hasta que el modelo está disponible en producción, quedara ordenado, fácil de reproducir y entendible para cualquier persona del equipo.

---

## Estructura del Proyecto

### 📂 `src/data`: 
En esta sección se concentra todo lo relacionado con los datos, tanto en su forma cruda como en su versión ya procesada. Es decir, aquí almacenamos los datos originales tal cual se reciben y también aquellos que han pasado por un primer nivel de limpieza y transformación.
- Punto de partida: se guardan los datasets de entrada o las conexiones directas a las fuentes de información.
- Procesamiento inicial: se incluyen los scripts que realizan el ETL básico, es decir, la extracción desde archivos o bases de datos, la limpieza preliminar y el guardado en formatos intermedios listos para usarse en etapas posteriores.
se definió un pipeline que garantiza que siempre recibamos datos en el mismo formato, sin duplicados ni columnas inconsistentes.

---

### 📂 `src/etl: 
En esta parte definimos los pasos de limpieza y transformación básica de los datos. El ETL funciona como un filtro que depura errores, inconsistencias o formatos extraños antes de que la información avance a la siguiente fase. Así aseguramos que todos trabajemos con datasets limpios y ordenados, evitando valores atípicos que podrían afectar el entrenamiento o las predicciones

Dentro de esta sección se encuentran los pipelines de transformación, que se encargan de:
- Elimina valores faltantes, corrige tipos de datos y estandariza formatos.  
- Asegura que cualquier dataset nuevo siga el mismo flujo.
- Garantizar que cualquier dataset nuevo siga exactamente el mismo flujo, evitando problemas futuros en las etapas de modelado o predicción.

---

### 📂 `src/feature_engineering`
En esta etapa dimos el salto de datos crudos a variables realmente útiles. Es decir, transformamos las columnas originales en información que el modelo puede aprovechar al máximo: codificaciones de texto, escalado de números y generación de nuevas variables. Esta parte fue clave porque aquí es donde los datos empezaron a tener verdadera “inteligencia”.
El módulo de feature engineering se encarga de:
  - Convertir variables categóricas en valores numéricos.  
  - Escalado de variables numéricas.  
  - Creación de nuevas variables predictivas.  
En resumen, aquí es donde los datos reciben una forma estructurada e inteligente, lista para que el modelo pueda realmente aprender y mejorar su desempeño  

---

### 📂 `src/models`
Este módulo es nuestro laboratorio de entrenamiento. Aquí pusimos a prueba distintos algoritmos, ajustamos hiperparámetros y evaluamos configuraciones hasta encontrar las que ofrecían mejores resultados. Fue, en pocas palabras, el espacio donde “nació” el modelo que hoy está listo para ponerse en producción.
Dentro de este módulo se concentra todo lo relacionado con el ciclo de entrenamiento de modelos:
- Se definen los algoritmos, hiperparámetros y el flujo completo de entrenamiento.
- Se implementan métricas de validación (accuracy, recall, F1, entre otras) para medir el desempeño.  
- Los modelos entrenados se exportan como artefactos listos para que la API los consuma de manera directa.

---

### 📂 `src/pipelines`
En este módulo lo que hicimos fue conectar todas las piezas en un flujo bien definido:
ETL → Feature Engineering → Entrenamiento → Validación.
Gracias a esto, el proceso se ejecuta siempre de manera ordenada y automática, sin depender de pasos manuales. Así garantizamos que, si mañana alguien del equipo corre el pipeline, obtenga exactamente el mismo resultado.
- Conecta todas las piezas en un flujo completo:
Este módulo se encarga de:  
- Integrar en un solo flujo las etapas de ETL, Feature Engineering, Modelado y Validación.
- Orquestar los scripts en el orden correcto para asegurar reproducibilidad.
- Hacer que el pipeline funcione igual tanto en desarrollo como en producción

---

### 📂 `src/app`
Este módulo es el encargado de tomar el modelo entrenado y exponerlo a través de una API. No se trata solo de devolver predicciones, sino de garantizar que todo el flujo sea confiable y consistente.
Nos enfocamos en que la API:
- Reciba peticiones externas y aplique el mismo preprocesamiento y devuelva predicciones listas para usar.
- Valide correctamente los datos de entrada, evitando errores o inconsistencias 
- Mantenga la trazabilidad mediante logging.
- Ofrezca endpoints adicionales de salud y versión del modelo, para asegurar disponibilidad y control de cambios.

---

### 📂 `src/tests`
En esta carpeta reunimos todas las pruebas automáticas que nos ayudan a validar que cada parte del proyecto funcione como debe. La idea fue asegurarnos de que tanto las piezas pequeñas como el sistema completo se mantengan estables, incluso cuando seguimos haciendo mejoras.
Incluye distintos tipos de pruebas:
  - **Unitarias** → verifican que cada función (ETL, feature engineering, etc.) haga lo que corresponde
  - **Integración** → validan que los pipelines se ejecuten de principio a fin sin problemas.  
  - **Contrato** → confirman que la API responda correctamente ante inputs válidos e inválidos. 
En resumen, este módulo nos da la confianza de que todo lo que construimos funciona igual hoy y seguirá funcionando mañana

---

### 📂 `src/config`
Por último, en esta carpeta centralizamos todas las configuraciones del proyecto: rutas, parámetros, nombres de artefactos y cualquier valor que pueda variar según el ambiente (desarrollo, pruebas o producción). De esta forma no tenemos que revisar el código línea por línea cada vez que haya que ajustar algo.
Aquí se incluyen:
- Configuraciones centralizadas: rutas, parámetros y variables de entorno.  
- Facilita cambiar setups entre desarrollo, pruebas y producción.  
- Mantiene el código limpio y ordenado.
Este módulo nos permite ajustar la configuración de manera simple y ordenada, manteniendo el código limpio y evitando errores por cambios manuales.
---

En resumen, este proyecto está organizado para cubrir todo el ciclo de vida de un modelo en producción, siguiendo un flujo completo de MLOps. Desde la entrada de datos crudos, su limpieza y transformación, hasta el entrenamiento, validación, despliegue mediante API y pruebas automáticas, cada módulo está diseñado para que el proceso sea reproducible, escalable y confiable:

## Resumen del Flujo
1. **Data + ETL** → Traer y limpiar datos.  
2. **Feature Engineering** → Transformarlos en variables útiles.  
3. **Models** → Entrenar, evaluar y guardar modelos.  
4. **Pipelines** → Orquestar todo el flujo.  
5. **App** → Exponer el modelo vía API.  
6. **Tests** → Garantizar calidad.  
7. **Config** → Mantener orden y flexibilidad.


## Definiciones Clave

### 🔹 ETL (Extract, Transform, Load)
Proceso para traer datos desde su origen, limpiarlos y guardarlos en un formato estructurado.  
- **Extract**: obtener datos desde archivos, BD o APIs.  
- **Transform**: limpiar nulos, duplicados y estandarizar formatos.  
- **Load**: guardar datos procesados listos para usar.  

### 🔹 Feature Engineering
Proceso para transformar datos crudos en variables valiosas para el modelo.  
- Codificación de categóricas.  
- Escalado numérico.  
- Creación de nuevas variables.  
- Selección de características más relevantes.  

