CONCENTRACIÓN DE MERCADO - Aplicación Streamlit (Organización Industrial)

Archivos
  app.py            Aplicación completa (indicadores, simulación, evaluador, interfaz)
  requirements.txt  Dependencias (streamlit, numpy, pandas, matplotlib)

Ejecución local
  1. Requisito: Python 3.9 o superior.
  2. (Opcional) Crear un entorno virtual:
        python -m venv venv
        source venv/bin/activate        (Windows: venv\Scripts\activate)
  3. Instalar dependencias:
        pip install -r requirements.txt
  4. Lanzar la aplicación:
        streamlit run app.py
  5. Abrir http://localhost:8501 si el navegador no se abre solo.

Despliegue en Streamlit Community Cloud
  Subir app.py y requirements.txt a un repositorio de GitHub, crear una nueva app
  en share.streamlit.io, elegir el repositorio y establecer "app.py" como archivo
  principal.

Uso rápido
  1. Elija indicador, N e iteraciones en el panel lateral.
  2. Defina el caso (cuotas manuales que sumen 1, o caso aleatorio).
  3. Responda al evaluador y lea la justificación técnica.
  4. Compare el caso con la distribución simulada (línea roja).
