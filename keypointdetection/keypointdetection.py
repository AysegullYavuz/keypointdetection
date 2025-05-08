import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

extracted_folder = input("Lütfen resimlerin bulunduğu klasörün yolunu girin: ")

obj_img_path = os.path.join(extracted_folder, 'cilek.jpg')  # Nesne resminizin adı
obj_img = cv2.imread(obj_img_path, cv2.IMREAD_GRAYSCALE)

# Referans resimleri yükle grayscale olarak
ref_img_paths = [
    os.path.join(extracted_folder, 'cilek2.jpg'),
    os.path.join(extracted_folder, 'cilek3.jpg'),
    os.path.join(extracted_folder, 'cilek4.jpg'),
    os.path.join(extracted_folder, 'cilek5.jpg'),
    os.path.join(extracted_folder, 'cilek6.jpg'),
    os.path.join(extracted_folder, 'cilek7.jpg'),
]
ref_imgs = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in ref_img_paths if os.path.exists(path)]

# Yeni yüklenen fotoğraf
new_img_path = input("Lütfen karşılaştırmak istediğiniz fotoğrafın yolunu girin: ")
new_img = cv2.imread(new_img_path, cv2.IMREAD_GRAYSCALE)

# SIFT Algoritmasını başlat
sift = cv2.SIFT_create()

# Nesne resminde keypoint ve deskriptörleri bulun
keypoints_obj, descriptors_obj = sift.detectAndCompute(obj_img, None)

# FLANN tabanlı matcher kullanarak eşleştirme
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Eşleşme için yapılacak iterasyon sayısı

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Eşik değeri ve eşleşme maskesi
MATCH_THRESHOLD = 0.7

def match_and_draw(img, keypoints, descriptors, title):
    matches = flann.knnMatch(descriptors_obj, descriptors, k=2)
    good_matches = [m for m, n in matches if m.distance < MATCH_THRESHOLD * n.distance]
    draw_params = dict(matchColor=(255, 0, 0), singlePointColor=None, flags=2)
    img_matches = cv2.drawMatches(obj_img, keypoints_obj, img, keypoints, good_matches, None, **draw_params)
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.imshow(img_matches)
    plt.show()

# Her referans resmi için eşleşmeyi göster
for i, ref_img in enumerate(ref_imgs):
    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_img, None)
    match_and_draw(ref_img, keypoints_ref, descriptors_ref, f'Çilek {i+2} ile Eşleşmeler')

# Yeni fotoğraf için keypoint ve deskriptörleri bulun ve karşılaştır
keypoints_new, descriptors_new = sift.detectAndCompute(new_img, None)
match_and_draw(new_img, keypoints_new, descriptors_new, 'Yeni Fotoğraf ile Eşleşmeler')
