import cv2

# Carga la imagen
imagen = cv2.imread('ejemplo.jpg')

# Convierte la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Crea un detector FAST
fast = cv2.FastFeatureDetector_create()

# Encuentra los puntos clave usando FAST
puntos_clave = fast.detect(imagen_gris, None)

# Dibuja los puntos clave en la imagen original
imagen_con_puntos_clave = cv2.drawKeypoints(imagen, puntos_clave, None, color=(255, 0, 0))

# Muestra la imagen con los puntos clave
cv2.imshow('Imagen con Puntos Clave FAST', imagen_con_puntos_clave)
cv2.waitKey(0)
cv2.destroyAllWindows()
