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
# Reverse the normalization so that dark areas represent more depth
depth_map_normalized = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX)

# Convert to uint8 format (color depth map)
depth_map_normalized = np.uint8(depth_map_normalized)

# Apply a colormap (for visualization purposes)
depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

# Display the depth map with color
cv2.imshow('Colored Depth Map', depth_map_colored)

# Save the output depth map to a file
cv2.imwrite('colored_depth_map_output.png', depth_map_colored)

# Wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
