
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def upscale_image(generator, lr_image_path, output_path, img_size):
    # Load the low-resolution image
    lr_img = load_img(lr_image_path, target_size=img_size)
    lr_img = img_to_array(lr_img) / 255.0  # Normalize to [0, 1]

    # Expand dimensions to match the input shape of the generator (batch_size, height, width, channels)
    lr_img = np.expand_dims(lr_img, axis=0)

    # Generate high-resolution image
    hr_img = generator.predict(lr_img)

    # Post-process the output
    hr_img = np.squeeze(hr_img, axis=0)  # Remove batch dimension
    hr_img = np.clip(hr_img, 0, 1)  # Ensure pixel values are in [0, 1]
    hr_img = (hr_img * 255.0).astype(np.uint8)  # Denormalize to [0, 255]

    # Convert array to image and save or display
    hr_img = array_to_img(hr_img)
    hr_img.save(output_path)
    return hr_img
for i in range(1,number_of_generators):
  # Load the pre-trained generator model
  generator = load_model(f'srgan_generator_epoch_{i}.keras')

  # Specify the low-resolution image and desired output path
  lr_image_path = 'image_path'
  output_path = f'high_res_image_{i}.png'

  # Define the size of the low-resolution input (e.g., 128x128)
  img_size = (128, 128)

  # Upscale the image
  high_res_image = upscale_image(generator, lr_image_path, output_path, img_size)

  # Display the high-resolution image
  high_res_image.show()
