
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca gambar RGB
gambar = iio.imread("gambar.jpg")

# Memastikan gambar memiliki 3 kanal warna RGB
gambar_rgb = gambar[:, :, :3]

# 2. Mengubah RGB menjadi grayscale
grayscale = (
    0.299 * gambar_rgb[:, :, 0] +
    0.587 * gambar_rgb[:, :, 1] +
    0.114 * gambar_rgb[:, :, 2]
).astype(np.uint8)

# 3. Menghitung histogram intensitas 0-255
histogram = np.bincount(grayscale.ravel(), minlength=256)

# 4. Menampilkan jumlah piksel setiap intensitas
print("Jumlah total piksel:", grayscale.size)
print("\nHistogram grayscale:")

for intensitas, jumlah in enumerate(histogram):
    print(f"Intensitas {intensitas}: {jumlah} piksel")

# 5. Menampilkan gambar asli, grayscale, dan histogram
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gambar_rgb)
plt.title("Gambar RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(grayscale, cmap="gray", vmin=0, vmax=255)
plt.title("Gambar Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.bar(range(256), histogram, width=1.0)
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas Piksel (0-255)")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)

# 6. Mencari intensitas yang paling dominan
intensitas_dominan = np.argmax(histogram)
jumlah_dominan = histogram[intensitas_dominan]

print("\nIntensitas paling dominan:", intensitas_dominan)
print("Jumlah piksel pada intensitas tersebut:", jumlah_dominan)

plt.tight_layout()
plt.show()