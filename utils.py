import cv2
import base64

def encode_image(image_path):
    """Read and encode image to base64"""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8') 