from PIL import Image
import os

dir_path = r"screenshots\USER"
image_paths = [
    os.path.join(dir_path, file)
    for file in os.listdir(dir_path)
    if file.endswith(("png", "jpg", "jpeg"))
][:120]


# Open images and store them in a list
images = [Image.open(image_path) for image_path in image_paths]

# Save as a GIF
images[0].save(
    "output.gif",
    save_all=True,
    append_images=images[1:],
    duration=100,  # Duration in milliseconds
    loop=0,  # Loop forever
)
