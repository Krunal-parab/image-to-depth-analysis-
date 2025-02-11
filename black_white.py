import torch
import cv2
import numpy as np
from torchvision import transforms

# Load MiDaS model (Pre-trained on large datasets)
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")

# Load the MiDaS model for different versions (v2 model is more accurate)
midas_model.eval()

# Define the transform for the input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(384),  # Resize to model input size
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Read the input image
image_path = 'krunal.jpeg'  # Replace with your image path
image = cv2.imread(image_path)

# Transform the image to match the model's input format
image_input = transform(image).unsqueeze(0)

# Run depth estimation with the model
with torch.no_grad():
    depth_map = midas_model(image_input)

# Post-process the depth map
depth_map = depth_map.squeeze().cpu().numpy()

# Normalize the depth map to fit into a visual range [0, 255]
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8 format (black and white image)
depth_map_normalized = np.uint8(depth_map_normalized)

# Display the depth map
cv2.imshow('Depth Map', depth_map_normalized)

# Save the output depth map to a file
cv2.imwrite('depth_map_output.png', depth_map_normalized)

# Wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
