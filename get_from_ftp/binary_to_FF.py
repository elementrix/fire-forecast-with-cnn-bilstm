
from PIL import Image
import numpy as np
import imageio

# 이미지 불러오기
img = Image.open('cropped_image1.png').convert('RGB')
img_np = np.array(img)

# R, G, B 분리
r = img_np[:, :, 0]
g = img_np[:, :, 1]
b = img_np[:, :, 2]

# 🔴 "순수한 빨간색" 조건 정의
# R은 높고, G와 B는 낮아야 함 (노란색 제거)
red_mask = (r > 150) & (g < 100) & (b < 100)

# 이진화 (정답 마스크 생성)
binary_mask = np.zeros_like(r, dtype=np.uint8)
binary_mask[red_mask] = 1  # 빨간색 → 1, 나머지 → 0

# 픽셀 수 계산
red_pixel_count = np.sum(binary_mask == 1)
total_pixel_count = binary_mask.size
red_ratio = red_pixel_count / total_pixel_count

# 출력
print(f"🔴 빨간색 픽셀 수: {red_pixel_count}")
print(f"🔳 전체 픽셀 수: {total_pixel_count}")
print(f"📊 비율: {red_ratio:.4f} ({red_ratio*100:.2f}%)")

# 정답 마스크 저장 (AI용)
# 1 → 255로 확장하여 저장
imageio.imwrite('ai_mask.png', binary_mask * 255)
print("✅ 정답 마스크 저장 완료: ai_mask.png")


