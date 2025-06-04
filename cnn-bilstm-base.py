import torch
import torch.nn as nn 
import torch.nn.functional as F
import os
import glob

from PIL import Image
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 1) StreamCNN: 각 입력 이미지 스트림을 처리하는 CNN 블록
class StreamCNN(nn.Module):
    def __init__(self, in_ch = 1, out_ch=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, in_ch, H, W)
        return self.block(x)  # → (B, out_ch, H, W)


# 2) FusionWithBiLSTM: 
#    8개 스트림 합친 뒤 채널 어텐션 → BiLSTM → Conv → 최종 1채널 예측
class FusionWithBiLSTM(nn.Module):
    def __init__(self, in_ch, mid_ch=128, lstm_hidden=64):
        super().__init__()

        # (1) 첫 번째 Conv: in_ch → mid_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        # (2) 채널 어텐션 (AdaptiveAvgPool2d → MLP → Sigmoid)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # → (B, mid_ch, 1, 1)
            nn.Conv2d(mid_ch, mid_ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 8, mid_ch, 1),
            nn.Sigmoid()                           # → (B, mid_ch, 1, 1)
        )

        # (3) BiLSTM: 입력 feature dim = mid_ch, hidden_size = lstm_hidden
        self.bilstm = nn.LSTM(
            input_size=mid_ch,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # (4) BiLSTM 뒤 Conv2d: (lstm_hidden*2 → mid_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(lstm_hidden * 2, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        # (5) 최종 1×1 Conv: mid_ch → 1
        self.conv_out = nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True)


    def forward(self, fused_feats):
        B, C, H, W = fused_feats.shape

        # 1) 첫 번째 Conv 블록
        x = self.conv1(fused_feats)       # → (B, mid_ch, H, W)

        # 2) 채널 어텐션 적용
        ca = self.channel_attn(x)          # → (B, mid_ch, 1, 1)
        x = x * ca                         # → (B, mid_ch, H, W)

        # 3) BiLSTM 처리: 
        #    (B, mid_ch, H, W) → (B, H, W, mid_ch) → (B*H, W, mid_ch)
        x_seq = x.permute(0, 2, 3, 1).contiguous()  # → (B, H, W, mid_ch)
        x_seq = x_seq.view(B * H, W, -1)            # → (B*H, W, mid_ch)

        # 4) BiLSTM 순전파 → (B*H, W, lstm_hidden*2)
        lstm_out, _ = self.bilstm(x_seq)

        # 5) 다시 (B, H, W, lstm_hidden*2) → (B, lstm_hidden*2, H, W)
        lstm_out = lstm_out.view(B, H, W, -1)       # → (B, H, W, lstm_hidden*2)
        lstm_feat = lstm_out.permute(0, 3, 1, 2).contiguous()  # → (B, 2*lstm_hidden, H, W)

        # 6) BiLSTM 출력 채널 → mid_ch로 맞추는 Conv
        x2 = self.conv2(lstm_feat)                # → (B, mid_ch, H, W)

        # 7) 최종 1×1 Conv → Sigmoid
        out = self.conv_out(x2)                   # → (B, 1, H, W)
        return torch.sigmoid(out)                 # → (B, 1, H, W)


# 3) 전체 모델: “8개 스트림 + FusionWithBiLSTM”
class EightStreamFireModelWithLSTM(nn.Module):
    def __init__(self, in_ch_list, stream_out_ch=32, mid_ch=128, lstm_hidden=64):
        super().__init__()
        assert len(in_ch_list) == 8, "입력 이미지 폴더는 8개여야 합니다."

        # 1) 8개 StreamCNN 생성
        self.streams = nn.ModuleList([
            StreamCNN(in_ch=in_ch_list[i], out_ch=stream_out_ch)
            for i in range(8)
        ])

        # 2) FusionWithBiLSTM 생성: in_ch = 8 * stream_out_ch
        self.fusion_lstm = FusionWithBiLSTM(
            in_ch = 8 * stream_out_ch,
            mid_ch = mid_ch,
            lstm_hidden = lstm_hidden
        )

    def forward(self, imgs):
        """
        imgs: 리스트 형태로 [img1, img2, ..., img8]
              각 img_i.shape == (B, C_i, H, W)
        """
        feats = []
        for i in range(len(imgs)):  
            # imgs[i]: (B, C_i, H, W)
            feat_i = self.streams[i](imgs[i])  # → (B, stream_out_ch, H, W)
            feats.append(feat_i)

        # 3) 채널 축 결합 → (B, 8*stream_out_ch, H, W)
        fused_feats = torch.cat(feats, dim=1)

        # 4) Fusion + BiLSTM → 최종 이진 마스크 확률 맵 (B, 1, H, W)
        return self.fusion_lstm(fused_feats)



class WildfireDataset(Dataset):
    """
    Dataset root: 모델과 동일 레밸 / Dataset 폴더 경로
    label: FF
    input: FF 제외 나머지 폴더 
    """

    def __init__(self, dataset_root: str,
                       label_folder_name: str = "FF",
                       input_folder_names: list = None,
                       transform=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.label_folder = os.path.join(dataset_root, label_folder_name)
        # FF 폴더 외 나머지 모두 입력 폴더로 간주
        if input_folder_names is None:
            # 자동으로 FF 제외하고 나머지 폴더명 가져오기
            all_dirs = sorted([
                d for d in os.listdir(dataset_root)
                if os.path.isdir(os.path.join(dataset_root, d))
            ])
            self.input_folders = [d for d in all_dirs if d != label_folder_name]
        else:
            self.input_folders = input_folder_names
        
        self.transform = transform if transform is not None else transforms.ToTensor()

        # 1) 라벨(FF) 폴더 내부에 있는 모든 이미지 파일명(확장자 포함)을 모아서 정렬
        #    예: ['202504032330.png', '202504032340.png', …]
        label_paths = sorted(glob.glob(os.path.join(self.label_folder, "*.*")))
        self.samples = []  # (input_paths_list, label_path) 튜플을 저장할 리스트

        for lbl_path in label_paths:
            fname = os.path.basename(lbl_path)             # 예: '202504032340.png'
            ts_str, ext = os.path.splitext(fname)          # ts_str='202504032340', ext='.png'

            # 2) 타임스탬프 문자열을 datetime 객체로 변환
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d%H%M")
            except ValueError:
                # 이름 형식이 yyyyMMddHHmm 에 맞지 않으면 스킵
                continue

            # 3) 10분 전 타임스탬프 계산
            prev_ts = ts - timedelta(minutes=10)
            prev_ts_str = prev_ts.strftime("%Y%m%d%H%M")   # 예: '202504032330'

            # 4) 각 입력 폴더에서 prev_ts_str 로 시작하는 파일이 있는지 확인
            input_paths = []
            missing_flag = False
            for in_folder in self.input_folders:
                folder_path = os.path.join(self.dataset_root, in_folder)
                # 아래 glob 은 prev_ts_str 와 어떤 확장자(.png, .jpg 등)도 매칭
                candidate = glob.glob(os.path.join(folder_path, prev_ts_str + ".*"))
                if len(candidate) == 0:
                    # 하나라도 없다면 이 샘플(Ff) 자체를 버림
                    missing_flag = True
                    break
                else:
                    # 여러 개가 나와도 첫 번째만 사용(.png/.jpg 중 하나겠지만)
                    input_paths.append(candidate[0])

            if missing_flag:
                continue

            # 5) 모두 존재하면 튜플로 추가
            #    ( [in_path1, in_path2, …], lbl_path )
            self.samples.append((input_paths, lbl_path))

        # 이제 self.samples 에 있는 수만큼 __len__ 이 결정됨
        # print("총 생성된 샘플 개수:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        """
        - self.samples[idx] = ( [in_path1, in_path2, …], label_path )
        - 입력 폴더 수만큼의 이미지와 라벨 이미지를 읽어서 Tensor 로 반환
        """
        input_paths, label_path = self.samples[idx]

        # 1) 입력 이미지들 로드
        inputs = []
        for p in input_paths:
            img = Image.open(p)
            img = img.convert("RGB") if img.mode != "RGB" else img  # 강제 RGB
            # 만약 Gray(1채널)로 쓰고 싶으면 .convert("L")로 바꿔도 됨
            tensor_img = self.transform(img)  # → (C, H, W)
            inputs.append(tensor_img)
        # inputs: 길이 N_list, 각각 (C_i, H, W)

        # 2) 라벨 이미지 로드 (FF)
        lbl = Image.open(label_path).convert("L")
        lbl_tensor = self.transform(lbl)      # → (1, H, W) 형태
        # (원한다면 0/1 이진 마스크로 변환)
        lbl_tensor = (lbl_tensor > 0.5).float()

        # 3) 반환
        #    inputs: [Tensor_1, Tensor_2, ..., Tensor_N]
        #    label:  Tensor (1, H, W)
        return inputs, lbl_tensor


def default_collate_fildfiew(batch):
    # 1) 입력 폴더 수 (첫 번째 샘플 기준)
    sample0_inputs, _ = batch[0]
    num_input_folders = len(sample0_inputs)

    # 2) 각 채널별로 모아서 배치 차원으로 쌓기
    #    e.g. channel별 리스트: [ [], [], … ]×num_input_folders
    channel_lists = [ [] for _ in range(num_input_folders) ]
    labels = []

    for inputs, lbl in batch:
        for c_idx, img_tensor in enumerate(inputs):
            channel_lists[c_idx].append(img_tensor)
        labels.append(lbl)

    # 3) channel_lists[c] 를 torch.stack 으로 쌓으면 (B, C_c, H, W)
    batch_inputs = [ torch.stack(channel_lists[c_idx], dim=0) 
                     for c_idx in range(num_input_folders) ]

    # 4) labels 리스트를 (B,1,H,W) 로 stack
    batch_labels = torch.stack(labels, dim=0)

    return batch_inputs, batch_labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3.1) Dataset 인스턴스 & Split ---
    dataset_root = "Dataset"
    # 자동으로 "FF" 제외 폴더를 입력 폴더로 잡음
    full_dataset = WildfireDataset(
        dataset_root=dataset_root,
        label_folder_name="FF",
        input_folder_names=None,
        transform=transforms.Compose([
            transforms.Resize((220, 550)),
            transforms.ToTensor()
        ])
    )

    total_len = len(full_dataset)
    train_len = int(total_len * 0.8)
    val_len   = int(total_len * 0.1)
    test_len  = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # --- 3.2) DataLoader 생성 ---
    batch_size = 4  # GPU 메모리에 맞춰 조정
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=wildfire_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=wildfire_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=wildfire_collate
    )

    print(f"Train/Val/Test 샘플 수: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    # 첫 배치에서 in_ch_list 자동 추출
    sample_inputs, _ = full_dataset[0]
    in_ch_list = [tensor.shape[0] for tensor in sample_inputs]  # 예: [3,1,3,3,1,1,3,3]

    model = EightStreamFireModelWithLSTM(
        in_ch_list=in_ch_list,
        stream_out_ch=32,
        mid_ch=128,
        lstm_hidden=64
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_labels in loader:
            # batch_inputs: list of 8 tensors, each (B, C_i, H, W)
            batch_inputs = [bi.to(device) for bi in batch_inputs]
            batch_labels = batch_labels.to(device)  # (B, 1, H, W)

            outputs = model(batch_inputs)           # (B, 1, H, W)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.size(0)

        return total_loss / len(loader.dataset)

    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in loader:
                batch_inputs = [bi.to(device) for bi in batch_inputs]
                batch_labels = batch_labels.to(device)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item() * batch_labels.size(0)

        return total_loss / len(loader.dataset)

    num_epochs = 20
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{num_epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        # 검증 손실이 좋아질 때만 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> Best Model 저장 완료")

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")