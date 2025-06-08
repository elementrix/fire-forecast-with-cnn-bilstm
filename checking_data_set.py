import os
import glob
from PIL import Image, UnidentifiedImageError
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 진행 상황 표시용

def check_image(image_path):
    """이미지 파일이 열리는지 확인하는 함수"""
    try:
        with Image.open(image_path) as img:
            # 단순히 열어보는 것보다 실제로 데이터 로드 시도
            img.load()
        return None  # 정상이면 None 반환
    except (UnidentifiedImageError, OSError, IOError) as e:
        return (image_path, str(e))  # 문제가 있으면 경로와 오류 메시지 반환

def scan_dataset_folder(dataset_root="Dataset", extensions=None):
    """Dataset 폴더의 모든 이미지 파일 검사"""
    start_time = time.time()
    print(f"검사 시작: {dataset_root} 폴더")
    
    # 기본 이미지 확장자 설정
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif']
    
    # 모든 하위 폴더 찾기
    all_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) 
               if os.path.isdir(os.path.join(dataset_root, d))]
    
    print(f"총 {len(all_dirs)}개 하위 폴더 발견: {', '.join(os.path.basename(d) for d in all_dirs)}")
    
    # 모든 이미지 파일 경로 수집
    all_images = []
    for folder in all_dirs:
        folder_name = os.path.basename(folder)
        folder_images = []
        
        for ext in extensions:
            images = glob.glob(os.path.join(folder, f"*{ext}"))
            folder_images.extend(images)
        
        all_images.extend(folder_images)
        print(f"  - {folder_name}: {len(folder_images)}개 이미지 파일")
    
    print(f"\n총 {len(all_images)}개 이미지 파일 검사 중...")
    
    # 이미지 파일 검사 (병렬 처리로 속도 향상)
    broken_images = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(check_image, all_images), total=len(all_images)))
    
    # 결과 필터링 (None이 아닌 결과만 수집)
    broken_images = [result for result in results if result is not None]
    
    # 결과 출력 및 파일로 저장
    elapsed_time = time.time() - start_time
    print(f"\n검사 완료! 소요 시간: {elapsed_time:.2f}초")
    print(f"총 {len(broken_images)}개의 손상된 이미지 파일 발견")
    
    # 폴더별 손상된 파일 통계
    if broken_images:
        folder_stats = {}
        for img_path, _ in broken_images:
            folder = os.path.dirname(img_path)
            folder_name = os.path.basename(folder)
            folder_stats[folder_name] = folder_stats.get(folder_name, 0) + 1
        
        print("\n폴더별 손상 파일 통계:")
        for folder, count in folder_stats.items():
            print(f"  - {folder}: {count}개")
            
        with open("broken_images.txt", "w") as f:
            f.write(f"# 손상된 이미지 파일 목록 (총 {len(broken_images)}개)\n")
            f.write(f"# 검사 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for img_path, error in broken_images:
                line = f"{img_path}: {error}"
                f.write(line + "\n")
        
        print(f"\n손상된 이미지 목록이 'broken_images.txt'에 저장되었습니다.")
    else:
        print("모든 이미지 파일이 정상입니다.")

if __name__ == "__main__":
    scan_dataset_folder("Dataset")