# Least_Squares_Project

# Least Squares Classifier 📊

## Descripción
Este proyecto implementa un clasificador basado en el método de mínimos cuadrados y su versión regularizada (Tikhonov/Ridge). 

El objetivo es analizar el comportamiento de ambos modelos en tareas de clasificación binaria y comparar su desempeño en términos de estabilidad y precisión.

---

## Fundamento Teórico

El modelo de mínimos cuadrados busca minimizar el error cuadrático:

||Ax - b||²

Mientras que la versión regularizada añade un término de penalización:

||Ax - b||² + λ||x||²

Esto permite:
- Reducir el sobreajuste
- Mejorar la estabilidad numérica
- Manejar datos con ruido

---

## Tecnologías usadas

- Python
- NumPy
- Matplotlib

---

## Estructura del proyecto
src/
│
├── least_squares.py # Implementación básica
├── regularized_ls.py # Implementación con regularización
└── main.py # Ejecución y comparación
ss