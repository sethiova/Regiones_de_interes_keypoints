import cv2
# Desde la versión 3.4.2 de OpenCV, el algoritmo SURF ya no se incluye de forma predeterminada debido a restricciones de patentes.

# Carga la imagen
imagen = cv2.imread('ejemplo.jpg')

# Convierte la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Crea un detector SURF
surf = cv2.SURF_create()

# Encuentra los puntos clave y los descriptores usando SURF
puntos_clave, descriptores = surf.detectAndCompute(imagen_gris, None)

# Dibuja los puntos clave en la imagen original
imagen_con_puntos_clave = cv2.drawKeypoints(imagen, puntos_clave, None)

# Muestra la imagen con los puntos clave
cv2.imshow('Imagen con Puntos Clave SURF', imagen_con_puntos_clave)
cv2.waitKey(0)
cv2.destroyAllWindows()
