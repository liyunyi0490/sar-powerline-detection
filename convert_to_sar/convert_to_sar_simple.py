import os
from PIL import Image
import random

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

input_folder = 'images'
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

def add_speckle_noise(image, L=20):
    """
    添加speckle噪声到图像
    L: 多视处理次数，值越大噪声越小，此处默认值为20
    """
    if image.mode != 'L':
        image = image.convert('L')
    
    width, height = image.size
    noisy_image = Image.new('L', (width, height))
    
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            gray = pixel / 255.0
            noise = random.gammavariate(L, 1.0/L)
            sar = gray * noise
            sar = max(0, min(1, sar))
            noisy_pixel = int(sar * 255)
            noisy_image.putpixel((x, y), noisy_pixel)
    
    return noisy_image

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)
    gray_image = image.convert('L')
    noisy_image = add_speckle_noise(gray_image, L=20)  # 可以根据需要更改L值
    output_path = os.path.join(output_folder, image_file)
    noisy_image.save(output_path)
    print(f"处理完成: {image_file}")

print("\n处理完成！所有图像已转换为灰度并添加了speckle噪声。")
