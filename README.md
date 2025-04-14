Repositorio en el que se utiliza Xception y EfficientNet, dos modelos preentrenados de detección de imágenes falsas generadas por otros modelos.
Primero se utilizan algunos scripts para la creación del dataset a utilizar y luego se "barajan" para conseguir la mayor homogeneidad posible en los fold.
Después, se ajustan algunos parámetros como la tasa de aprendizaje, el número de folds, el tamaño de lote, que depende de la capacidad del hardware y la semilla.
