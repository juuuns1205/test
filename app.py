
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pretty_midi
import glob
import os
#하나의 노트로 다음 노트를 예측
# MIDI 파일이 있는 디렉터리 경로
midi_dir = r"D:\maestro-v3.0.0"  # 여기에 MIDI 파일이 있는 폴더 경로 입력

# 'name_1.mid', 'name_2.mid' 순서대로 정렬된 파일 목록 가져오기
midi_files = sorted(glob.glob(os.path.join(midi_dir, "2006","*.midi"),recursive=True))

# MIDI 파일들을 순서대로 읽기
midi_data_list = []
for midi_file in midi_files:
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        midi_data_list.append(midi_data)
        print(f"Loaded: {midi_file}")
    except Exception as e:
        print(f"Failed to load {midi_file}: {e}")


# MIDI 데이터를 벡터로 변환하는 함수
def extract_features(midi_data):
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # 드럼 제외
            for note in instrument.notes:
                notes.append([note.pitch, note.start, note.end, note.velocity])
    
    if len(notes) == 0:
        return None  # 빈 MIDI는 제외
    
    return np.array(notes)

# 모든 MIDI 데이터를 벡터화
X_train = []
for midi in midi_data_list:
    features = extract_features(midi)
    if features is not None:
        X_train.append(features)

# NumPy 배열로 변환
X_train = np.vstack(X_train)  # (총 노트 수, 4) 형태
 # 예: (10000, 4)


#2. 간단한 신경망 모델 정의하기

# PyTorch 데이터 변환
X_train = torch.tensor(X_train[:, :3], dtype=torch.float32)  # 입력 (Pitch, Start, End)
y_train = torch.tensor(X_train[:, 0], dtype=torch.float32).clone().detach().view(-1, 1)  # 타겟 (다음 Pitch 예측)
print("Feature Shape:", y_train.shape) 
# 신경망 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # 입력 3개 → 16개 뉴런
        self.fc2 = nn.Linear(16, 8)  # 16개 뉴런 → 8개
        self.fc3 = nn.Linear(8, 1)   # 최종 출력 (다음 Pitch 예측)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 초기화
model = SimpleNN()
loss_function = nn.MSELoss()  # 평균 제곱 오차 (MSE)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # SGD (경사하강법)

print(model)  # 모델 구조 출력

num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()  # 기울기 초기화
    #predictions = model(X_train)  # 순전파 (Forward)
    predictions = model.forward(X_train)
    loss = loss_function(predictions, y_train)  # 손실 계산
    loss.backward()  # 역전파 (Backpropagation)
    optimizer.step()  # 가중치 업데이트 (Gradient Descent)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 새로운 입력 데이터를 생성하여 예측
test_input = torch.tensor([[60, 0.0, 0.5]], dtype=torch.float32)  # (Pitch=60, Start=0.0, End=0.5)
predicted_pitch = model(test_input).item()
print(f"Predicted Next Pitch: {predicted_pitch:.2f}")




