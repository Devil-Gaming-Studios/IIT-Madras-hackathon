from PIL import Image
import numpy as np

# Open the image
img = Image.open('robot.jpeg')
img_array = np.array(img)
h, w = img_array.shape[:2]

# Sample background color from corners
corners = [
    img_array[0, 0, :3],
    img_array[0, w-1, :3],
    img_array[h-1, 0, :3],
    img_array[h-1, w-1, :3]
]
bg_color = np.mean(corners, axis=0).astype(int)
print(f"Detected background color: RGB{tuple(bg_color)}")

# Convert to RGBA
if img.mode != 'RGBA':
    img = img.convert('RGBA')

img_array = np.array(img)

# Remove background by color-keying
threshold = 60
for y in range(h):
    for x in range(w):
        r, g, b = img_array[y, x, :3]
        dr = int(r) - bg_color[0]
        dg = int(g) - bg_color[1]
        db = int(b) - bg_color[2]
        dist = (dr*dr + dg*dg + db*db) ** 0.5
        
        if dist <= threshold:
            img_array[y, x, 3] = 0  # Make transparent

# Save as PNG
img = Image.fromarray(img_array)
img.save('robot.png')
print('✓ Background removed! Saved as robot.png')
