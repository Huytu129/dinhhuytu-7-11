import cv2
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Đọc ảnh vệ tinh
image = cv2.imread("a-nh-ma-n-hi-nh-2023-07-18-lu-1372-9168-1689662616.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển ảnh sang RGB để hiển thị đúng màu

pixels = image_rgb.reshape((-1, 3))

# Phân cụm bằng K-means
kmeans = KMeans(n_clusters=3)
kmeans.fit(pixels)
kmeans_labels = kmeans.predict(pixels)
kmeans_image = kmeans_labels.reshape(image_rgb.shape[:2])

# Phân cụm bằng FCM
pixels = pixels.T  # Chuyển mảng pixel cho FCM
n_clusters = 3
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    pixels, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

# Tìm cụm với độ thành viên cao nhất
fcm_labels = np.argmax(u, axis=0)
fcm_image = fcm_labels.reshape(image_rgb.shape[:2])

# Hiển thị 3 ảnh trong một cửa sổ
plt.figure(figsize=(15, 5))

# Hiển thị ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

# Hiển thị kết quả K-means
plt.subplot(1, 3, 2)
plt.imshow(kmeans_image, cmap="gray")  # Cmap tối
plt.title("K-means Clustering (Dark Mode)")
plt.axis('off')

# Hiển thị kết quả FCM
plt.subplot(1, 3, 3)
plt.imshow(fcm_image, cmap="gray")  # Cmap tối
plt.title("Fuzzy C-means Clustering (Dark Mode)")
plt.axis('off')

# Hiển thị kết quả
plt.tight_layout()
plt.show()
