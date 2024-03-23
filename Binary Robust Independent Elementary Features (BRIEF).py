import cv2

# Cargar la imagen
imagen = cv2.imread('ejemplo.jpg', cv2.COLOR_BGR2GRAY)

# Inicializar el detector ORB
orb = cv2.ORB_create()

# Encontrar puntos clave y descriptores con ORB
keypoints, descriptors = orb.detectAndCompute(imagen, None)

# Dibujar puntos clave en la imagen
img_keypoints = cv2.drawKeypoints(imagen, keypoints, None, color=(0,255,0), flags=0)

# Mostrar la imagen con puntos clave
cv2.imshow('Características ORB', img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
