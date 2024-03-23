import cv2

# Carga la imagen
imagen = cv2.imread('ejemplo.jpg')

# Convierte la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Crea un detector SIFT
sift = cv2.SIFT_create()

# Encuentra los puntos clave y los descriptores usando SIFT
puntos_clave, descriptores = sift.detectAndCompute(imagen_gris, None)

# Dibuja los puntos clave en la imagen original
imagen_con_puntos_clave = cv2.drawKeypoints(imagen, puntos_clave, None)

# Muestra la imagen con los puntos clave
cv2.imshow('Imagen con Puntos Clave SIFT', imagen_con_puntos_clave)
cv2.waitKey(0)
cv2.destroyAllWindows()
