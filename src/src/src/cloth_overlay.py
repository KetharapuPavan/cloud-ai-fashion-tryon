import cv2

def overlay_cloth(image, cloth_texture):
    blended = cv2.addWeighted(image, 0.8, cloth_texture, 0.2, 0)
    return blended
