library(magrittr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(corrplot)
library(lmtest)
library(glmnet)
library(caret)
library(yardstick)
library(nnet)

#1. Importar el dataset y ver cantidad de datos
dataset <- read.csv("https://raw.githubusercontent.com/SergioGerman412/Datasets/main/USA_Housing.csv", header = TRUE)
cat("La cantidad de columnas es", ncol(dataset), "y sus nombres son:", paste(names(dataset), collapse = ", "), "\n");
cat("Número de filas:", nrow(dataset), "\n");


#2. Se divide el dataset para el análisis y evitar tocar datos no vistos (datos prueba)
set.seed(123);
dato_entrenamiento <- dataset[sample(nrow(dataset), 0.8 * nrow(dataset)), ]
dato_prueba <- dataset[-seq_len(nrow(dato_entrenamiento)), ]

dato_entrenamiento_logistica <- data.frame(dato_entrenamiento)
dato_prueba_logistica <- data.frame(dato_prueba)


#3 Verifica si hay datos nulos en el dataset original (solo verificar)
cantidad_nulos <- sum(is.na(dataset))
cat("La cantidad de datos nulos es:", cantidad_nulos, "\n")

#4. Verifica duplicados en el dataset original (solo verificar)
duplicados <- dataset[duplicated(dataset), ]
cat("La cantidad de duplicados es:", nrow(duplicados), "\n")


###################################Se comienza el análisis##################

#5. Calcula estadísticas descriptivas
descriptive_stats <- summary(dato_entrenamiento)
print(descriptive_stats)


#6. boxplot para identificar outliers
set1 <- c("Avg..Area.House.Age", "Avg..Area.Number.of.Bedrooms", "Avg..Area.Number.of.Rooms")  
dato_entrenamiento %>%
  select(any_of(set1)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Valor") %>%
  ggplot(aes(x = Variable, y = Valor, fill = Variable)) +
  geom_boxplot() +
  labs(title = "Boxplot de Variables Numéricas",
       x = "Variable", y = "Valor") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.8),
        axis.text.x = element_text(angle = 45, hjust = 1))

set2 <- c("Area.Population", "Avg..Area.Income")  
dato_entrenamiento %>%
  select(any_of(set2)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Valor") %>%
  ggplot(aes(x = Variable, y = Valor, fill = Variable)) +
  geom_boxplot() +
  labs(title = "Boxplot de Variables Numéricas",
       x = "Variable", y = "Valor") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.8),
        axis.text.x = element_text(angle = 45, hjust = 1))


##7. Histograma teniendo en cuenta Regla de Scott para definir bindwidth y prueba de normalidad  Shapiro-Wilk

binwidth_scott <- function(x) {
  3.5 * sd(x) / length(x)^(1/3)
}

#Definir la función para crear histogramas
crear_histograma <- function(data, variable) {
  ggplot(data, aes(x = !!sym(variable))) +
    geom_histogram(binwidth = binwidth_scott(data[[variable]]),
                   fill = "blue", color = "black", alpha = 0.7) +
    labs(title = paste("Histograma de", variable),
         x = "Valor", y = "Frecuencia") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}


crear_histograma(dato_entrenamiento, "Avg..Area.Income")
crear_histograma(dato_entrenamiento, "Avg..Area.House.Age")
crear_histograma(dato_entrenamiento, "Avg..Area.Number.of.Rooms")
crear_histograma(dato_entrenamiento, "Avg..Area.Number.of.Bedrooms")
crear_histograma(dato_entrenamiento, "Area.Population")
crear_histograma(dato_entrenamiento, "Price")

shapiro.test(dato_entrenamiento$Avg..Area.Income)
shapiro.test(dato_entrenamiento$Avg..Area.House.Age)
shapiro.test(dato_entrenamiento$Avg..Area.Number.of.Rooms)
shapiro.test(dato_entrenamiento$Avg..Area.Number.of.Bedrooms)
shapiro.test(dato_entrenamiento$Area.Population)
shapiro.test(dato_entrenamiento$Price)


##8. Diagrama de dispersión

crear_diagrama_dispersión <- function(dataset, x_variable, y_variable) {
  ggplot(dataset, aes_string(x = x_variable, y = y_variable)) +
    geom_point() +
    labs(title = paste("Diagrama de Dispersión:", x_variable, "vs", y_variable),
         x = x_variable, y = y_variable) +
    theme_minimal()
}

crear_diagrama_dispersión(dato_entrenamiento, "Avg..Area.Income", "Price")
crear_diagrama_dispersión(dato_entrenamiento, "Avg..Area.House.Age", "Price")
crear_diagrama_dispersión(dato_entrenamiento, "Avg..Area.Number.of.Rooms", "Price")
crear_diagrama_dispersión(dato_entrenamiento, "Avg..Area.Number.of.Bedrooms", "Price")
crear_diagrama_dispersión(dato_entrenamiento, "Area.Population", "Price")


#9. Matriz de correlación
dato_entrenamiento <- subset(dato_entrenamiento, select = -c(Address))
matriz_correlacion <- cor(dato_entrenamiento)
par(mar = c(1, 1, 1, 1))
corrplot(matriz_correlacion, method = "color", type = "full", tl.col = "black", tl.srt = 45, addCoef.col = "black")

dato_prueba <- subset(dato_prueba, select = -c(Address))




###################################MODELO DE REGRESION LINEAL CON Avg..Area.Income##################

dato_modelo1 <- subset(dato_entrenamiento, select = c("Avg..Area.Income", "Price"))
matriz_correlacion_modelo1 <- cor(dato_modelo1)
print(matriz_correlacion_modelo1)

#10. Algoritmo de regresión lineal y evaluación del modelo


X <- as.matrix(dato_entrenamiento[, c("Avg..Area.Income")])
y <- dato_entrenamiento$Price
X <- cbind(X, rep(1, nrow(X)))


# Ajustar un modelo de regresión lineal simple
modelo_lineal <- lm(Price ~ Avg..Area.Income, data = dato_entrenamiento)
predicciones <- predict(modelo_lineal, newdata = dato_prueba)
# Mostrar un resumen del modelo 
summary(modelo_lineal)


# Obtener los residuos del modelo
residuos <- residuals(modelo_lineal)
# Graficar los residuos vs valores ajustados
plot(fitted(modelo_lineal), residuos, 
     xlab = "Valores Ajustados", 
     ylab = "Residuos",
     main = "Gráfico de Residuos vs Valores Ajustados")

MAE <- mean(abs(dato_prueba$Price - predicciones))
print(MAE)

###################################MODELO DE REGRESION MULTILINEAL##################

#11. Algoritmo de Regresión multilineal y evaluación del modelo


modelo_regresion_multilineal <- lm(Price ~ Avg..Area.House.Age + Avg..Area.Number.of.Rooms + Avg..Area.Number.of.Bedrooms + Area.Population + Avg..Area.Income, data = dato_entrenamiento)
predicciones <- predict(modelo_regresion_multilineal, newdata = dato_prueba)
summary(modelo_regresion_multilineal)


mae <- mean(abs(predicciones - dato_prueba$Price))
print(mae)


###################################MODELO DE REGRESION LOGISTICA##################

dato_entrenamiento_logistica <- dato_entrenamiento_logistica[, -which(names(dato_entrenamiento_logistica) == "Address")]
dato_prueba_logistica <- dato_prueba_logistica[, -which(names(dato_prueba_logistica) == "Address")]

#12. Algoritmo de Regresión Logística (algoritmo de clasificación)
puntos_corte <- c(0, 50000, 600000, 900000, Inf)

# Crear categoría categórica Price_Categoria
categorias_entrenamiento <- cut(dato_entrenamiento_logistica$Price, breaks = puntos_corte, labels = c("Bajo", "Medio", "Alto", " Muy Alto"), include.lowest = TRUE)
dato_entrenamiento_logistica$Price_Categoria <- categorias_entrenamiento
categorias_prueba <- cut(dato_prueba_logistica$Price, breaks = puntos_corte, labels = c("Bajo", "Medio", "Alto", " Muy Alto"), include.lowest = TRUE)
dato_prueba_logistica$Price_Categoria <- categorias_prueba

# Verificar el balanceo de clases
table(dato_entrenamiento_logistica$Price_Categoria)
table(dato_prueba_logistica$Price_Categoria)

# Crear la matriz de correlación para este nuevo modelo
matriz_cor2 <- cor(dato_entrenamiento_logistica[variables3])
print(matriz_cor2)


# Ajustar el modelo de regresión logística multinomial
modelo_logistico <- multinom(Price_Categoria ~ Avg..Area.Income + Avg..Area.House.Age + Avg..Area.Number.of.Rooms + Avg..Area.Number.of.Bedrooms + Area.Population, 
                             data = dato_entrenamiento_logistica)
summary(modelo_logistico)

# Realizar predicciones en el conjunto de prueba
predicciones <- predict(modelo_logistico, newdata = dato_prueba_logistica)
predicciones_numericas <- as.numeric(predicciones) - 1


# Crear la matriz de confusión
matriz_confusion <- confusionMatrix(data = predicciones, reference = dato_prueba_logistica$Price_Categoria)
print(matriz_confusion)

