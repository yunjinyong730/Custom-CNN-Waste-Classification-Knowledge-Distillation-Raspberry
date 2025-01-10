# 쓰레기 분류 Custom CNN 모델 Knowledge Distillation on Raspberry

## Custom CNN 모델을 쓰레기를 분류하는 모델 학습

- Model Architecture

```kotlin
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(waste_types), activation='softmax')
])
```

- Confusion Maxtrix

<img width="909" alt="스크린샷 2025-01-05 오후 8 35 00" src="https://github.com/user-attachments/assets/8431bec6-34d4-4bcc-936a-22c8b7cbeb50" />

- **cardboard**: 74개 정확히 예측, 28개 잘못 예측
- **paper**: 가장 높은 정확도를 보이는 것으로 보임(오류가 거의 없음)
- 모델이 **paper**와 같은 일부 카테고리에서 매우 높은 정확도를 보임
- 주요 분류 경향이 **대각선**에 분포되어 있으므로 전반적으로 분류 성능이 괜찮아 보임
- **cardboard**, **glass**, **plastic**에서 혼동이 발생하는 경우가 있음

### 학습 코드
<details>
<summary>training code 상세</summary>
<br>
  
```python
import os
import random
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# 경로 및 매개변수 정의
DATASET_DIR = "dataset"  # 원본 데이터셋 경로
OUTPUT_DIR = "data"  # 학습/검증/테스트 데이터셋이 저장될 경로
IMG_SIZE = (128, 128)  # 이미지 크기 (너비, 높이)
BATCH_SIZE = 32  # 배치 크기
EPOCHS = 20  # 학습 횟수

# 학습(train), 검증(valid), 테스트(test) 폴더 생성
subsets = ['train', 'valid', 'test']  # 데이터 분리 기준
waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # 분류할 쓰레기 타입

# 각 데이터셋 경로 생성
for subset in subsets:
    for waste_type in waste_types:
        folder = os.path.join(OUTPUT_DIR, subset, waste_type)
        os.makedirs(folder, exist_ok=True)  # 폴더가 없으면 생성

# 데이터를 train, valid, test로 나누는 함수
random.seed(42)  # 난수 시드 설정 (결과 재현 가능)
def split_indices(folder, train_ratio=0.5, valid_ratio=0.25):
    all_files = os.listdir(folder)  # 폴더 내 모든 파일 리스트 가져오기
    n = len(all_files)  # 전체 파일 수
    train_size = int(train_ratio * n)  # 학습 데이터 크기
    valid_size = int(valid_ratio * n)  # 검증 데이터 크기

    random.shuffle(all_files)  # 파일 순서를 랜덤하게 섞음
    train_files = all_files[:train_size]  # 학습 파일
    valid_files = all_files[train_size:train_size + valid_size]  # 검증 파일
    test_files = all_files[train_size + valid_size:]  # 테스트 파일

    return train_files, valid_files, test_files

# 각 쓰레기 타입에 대해 데이터를 나누고 파일 복사
for waste_type in waste_types:
    source_folder = os.path.join(DATASET_DIR, waste_type)  # 원본 폴더 경로
    train_files, valid_files, test_files = split_indices(source_folder)  # 파일 분할

    # train, valid, test 데이터셋에 파일 복사
    for subset, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        dest_folder = os.path.join(OUTPUT_DIR, subset, waste_type)
        for file in files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))

# 학습 및 검증을 위한 데이터 생성기
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.25)  # 이미지 값을 0~1 범위로 스케일링

# 학습 데이터 생성기
train_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'train'),  # 학습 데이터 경로
    target_size=IMG_SIZE,  # 이미지 크기 조정
    batch_size=BATCH_SIZE,  # 배치 크기
    class_mode='categorical',  # 다중 클래스 분류
)

# 검증 데이터 생성기
valid_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'valid'),  # 검증 데이터 경로
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # 합성곱 레이어
    tf.keras.layers.MaxPooling2D((2, 2)),  # 최대 풀링 레이어
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),  # 데이터를 1차원으로 변환
    tf.keras.layers.Dense(128, activation='relu'),  # 완전 연결 레이어
    tf.keras.layers.Dense(len(waste_types), activation='softmax')  # 출력 레이어 (다중 클래스)
])

# 모델 컴파일
model.compile(optimizer='adam',  # 옵티마이저
              loss='categorical_crossentropy',  # 손실 함수
              metrics=['accuracy'])  # 평가 지표

# 모델 학습
history = model.fit(
    train_generator,  # 학습 데이터
    epochs=EPOCHS,  # 학습 횟수
    validation_data=valid_generator  # 검증 데이터
)

# 모델 저장
model.save("waste_classifier.h5")

# 테스트 데이터셋을 위한 생성기
test_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'test'),  # 테스트 데이터 경로
    target_size=IMG_SIZE,
    batch_size=1,  # 한 번에 하나의 이미지만 사용
    class_mode='categorical',
    shuffle=False  # 결과 분석을 위해 순서 고정
)

# 모델 예측
y_pred = model.predict(test_generator)  # 테스트 데이터 예측
y_pred_classes = np.argmax(y_pred, axis=1)  # 예측 클래스 추출
y_true = test_generator.classes  # 실제 클래스

# 혼동 행렬 생성 및 시각화
cm = confusion_matrix(y_true, y_pred_classes)  # 혼동 행렬 계산
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=waste_types, yticklabels=waste_types)  # 히트맵
plt.xlabel('Predicted')  # x축 레이블
plt.ylabel('Actual')  # y축 레이블
plt.title('Confusion Matrix')  # 제목
plt.show()

# 분류 보고서 출력
print(classification_report(y_true, y_pred_classes, target_names=waste_types))

# 개별 테스트 이미지 시각화 및 예측
def predict_and_visualize(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)  # 이미지 로드 및 크기 조정
    image_array = img_to_array(image) / 255.0  # 배열로 변환 및 스케일링
    image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가

    prediction = model.predict(image_array)  # 예측
    predicted_class = waste_types[np.argmax(prediction)]  # 예측 클래스

    plt.imshow(image)  # 이미지 시각화
    plt.title(f"Predicted: {predicted_class}")  # 예측 결과 표시
    plt.axis('off')  # 축 제거
    plt.show()

# 랜덤한 테스트 이미지 5개 시각화 및 예측
test_images = test_generator.filenames  # 테스트 이미지 파일 경로
for i in range(5):  # 5개 이미지
    test_image_path = os.path.join(test_generator.directory, test_images[i])  # 이미지 전체 경로
    predict_and_visualize(test_image_path)  # 예측 및 시각화

```
- 데이터 준비: 데이터 분리 및 전처리
- 모델 구성: CNN 모델 구축
- 학습 및 평가: 모델 학습 및 성능 평가
- 예측 및 시각화: 테스트 데이터 예측 및 결과 시각화

</details>



### 실제 테스트
<details>
<summary>실제 테스트 결과 (성공사례)</summary>
<br>
<img width="636" alt="스크린샷 2025-01-05 오후 9 40 14" src="https://github.com/user-attachments/assets/add967be-3e1e-413d-b7be-e770c5de1602" />
<img width="638" alt="스크린샷 2025-01-05 오후 9 38 33" src="https://github.com/user-attachments/assets/11abb8d2-7a26-4037-84d5-60e9907aaa48" />
<img width="637" alt="스크린샷 2025-01-05 오후 9 36 51" src="https://github.com/user-attachments/assets/18aee897-1ebd-4d59-937c-b53112f73d46" />
<img width="637" alt="스크린샷 2025-01-05 오후 9 35 28" src="https://github.com/user-attachments/assets/e991cbd5-66d1-47ee-819d-a8ef5ec4efc0" />
<img width="639" alt="스크린샷 2025-01-05 오후 9 34 03" src="https://github.com/user-attachments/assets/4fd2fa7f-38ee-49ea-947e-e1ebf6ae4c61" />
<img width="639" alt="스크린샷 2025-01-05 오후 9 32 45" src="https://github.com/user-attachments/assets/11064a84-5bcb-4d95-a83b-f265428bbc15" />
<img width="642" alt="스크린샷 2025-01-05 오후 9 15 30" src="https://github.com/user-attachments/assets/23c893f7-08ed-471d-bc0f-2710f50503de" />
<img width="634" alt="스크린샷 2025-01-05 오후 9 15 20" src="https://github.com/user-attachments/assets/7eec9fdc-62b5-4a6b-ab14-d7a8adf56124" />
<img width="632" alt="스크린샷 2025-01-05 오후 9 15 00" src="https://github.com/user-attachments/assets/762474d8-d100-4369-9451-015c025813a7" />
</details>

# 경량 CNN 모델을 student 모델로 knowledge Distillation 수행

```python
// teacher 모델

# Define the teacher model
teacher_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(waste_types), activation='softmax')
])
```

```python
// student 모델

# Define the student model
student_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(waste_types), activation='softmax')
])
```

### knowledge distillation 과정 전체 코드

<details>
<summary>KD 전체 코드</summary>
<br>

 폐기물 이미지 데이터를 처리하고, 교사 모델(Teacher Model)과 학생 모델(Student Model)을 사용해 **지식 증류(Knowledge Distillation)** 방식으로 폐기물 분류기를 학습한다

 ```python
import os
import random
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# 경로 및 파라미터 정의
DATASET_DIR = "dataset"  # 원본 데이터셋 디렉토리
OUTPUT_DIR = "data"  # 훈련, 검증, 테스트 데이터를 저장할 디렉토리
IMG_SIZE = (128, 128)  # 이미지 크기 (128x128 픽셀)
BATCH_SIZE = 32  # 배치 크기
EPOCHS = 20  # 학습 에폭 수
MAX_STEPS_PER_EPOCH = 1000  # 에폭 당 최대 스텝 수

# train, valid, test 디렉토리를 생성
subsets = ['train', 'valid', 'test']  # 데이터셋 분할: 훈련, 검증, 테스트
waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # 쓰레기 종류

for subset in subsets:
    for waste_type in waste_types:
        folder = os.path.join(OUTPUT_DIR, subset, waste_type)  # 하위 디렉토리 생성
        os.makedirs(folder, exist_ok=True)

# 데이터를 train, valid, test로 나누기
random.seed(42)  # 랜덤 시드 고정
def split_indices(folder, train_ratio=0.5, valid_ratio=0.25):
    all_files = os.listdir(folder)  # 폴더 내 모든 파일 이름 가져오기
    n = len(all_files)
    train_size = int(train_ratio * n)  # 훈련 데이터 크기 계산
    valid_size = int(valid_ratio * n)  # 검증 데이터 크기 계산

    random.shuffle(all_files)  # 파일 리스트 섞기
    train_files = all_files[:train_size]
    valid_files = all_files[train_size:train_size + valid_size]
    test_files = all_files[train_size + valid_size:]

    return train_files, valid_files, test_files

for waste_type in waste_types:
    source_folder = os.path.join(DATASET_DIR, waste_type)  # 원본 데이터 폴더
    train_files, valid_files, test_files = split_indices(source_folder)

    for subset, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        dest_folder = os.path.join(OUTPUT_DIR, subset, waste_type)  # 대상 폴더
        for file in files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))  # 파일 복사

# 훈련 및 검증용 데이터 제너레이터 생성
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.25)  # 이미지 정규화 및 검증 비율 설정

train_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

valid_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

# Teacher 모델 정의
teacher_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Conv2D 레이어
    tf.keras.layers.MaxPooling2D((2, 2)),  # MaxPooling2D
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),  # 특징 벡터 변환
    tf.keras.layers.Dense(128, activation='relu'),  # Fully Connected Layer
    tf.keras.layers.Dense(len(waste_types), activation='softmax')  # 클래스 개수만큼 출력
])

teacher_model.compile(optimizer='adam',  # Adam 옵티마이저
              loss='categorical_crossentropy',  # 손실 함수
              metrics=['accuracy'])  # 평가 지표

# Teacher 모델 학습
teacher_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator
)

teacher_model.save("waste_teacher_model.h5")  # Teacher 모델 저장

# Student 모델 정의
student_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(waste_types), activation='softmax')
])

# Knowledge Distillation 손실 함수 정의
def distillation_loss(y_true, y_pred, teacher_logits, temperature=5):
    soft_targets = tf.nn.softmax(teacher_logits / temperature)  # Teacher 출력 소프트맥스
    student_logits = tf.nn.softmax(y_pred / temperature)  # Student 출력 소프트맥스
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(soft_targets, student_logits))  # 손실 계산
    return loss

# Student 모델 컴파일
student_optimizer = tf.keras.optimizers.Adam()
student_model.compile(
    optimizer=student_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Knowledge Distillation 학습
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(train_generator):
        if step >= MAX_STEPS_PER_EPOCH:  # 최대 스텝 제한
            print(f"Reached max steps per epoch: {MAX_STEPS_PER_EPOCH}")
            break

        x_batch, y_batch = batch  # 배치 데이터
        teacher_logits = teacher_model(x_batch, training=False)  # Teacher 예측값

        with tf.GradientTape() as tape:
            student_logits = student_model(x_batch, training=True)  # Student 예측값
            loss = distillation_loss(y_batch, student_logits, teacher_logits)  # Knowledge Distillation 손실 계산

        grads = tape.gradient(loss, student_model.trainable_weights)  # 기울기 계산
        student_optimizer.apply_gradients(zip(grads, student_model.trainable_weights))  # 가중치 업데이트

        # 진행 상황 출력
        if step % 100 == 0:  # 100 스텝마다 출력
            print(f"Step {step}, Loss: {loss.numpy():.4f}")

    # 검증 데이터 평가
    val_loss, val_accuracy = student_model.evaluate(valid_generator, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Student 모델 저장
student_model.save("waste_student_model.keras")

# Student 모델 평가
student_model.evaluate(valid_generator)

# 혼동 행렬 및 분류 보고서 출력
test_generator = data_gen.flow_from_directory(
    os.path.join(OUTPUT_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

y_pred = student_model.predict(test_generator)  # 테스트 데이터 예측
y_pred_classes = np.argmax(y_pred, axis=1)  # 예측된 클래스
y_true = test_generator.classes  # 실제 클래스

cm = confusion_matrix(y_true, y_pred_classes)  # 혼동 행렬 생성
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=waste_types, yticklabels=waste_types)  # 히트맵 시각화
plt.xlabel('Predicted')  # x축 라벨
plt.ylabel('Actual')  # y축 라벨
plt.title('Confusion Matrix')  # 제목
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=waste_types))  # 분류 보고서 출력

# 테스트 이미지 예측 및 시각화 함수
def predict_and_visualize(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)  # 이미지 로드 및 크기 조정
    image_array = img_to_array(image) / 255.0  # 배열 변환 및 정규화
    image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가

    prediction = student_model.predict(image_array)  # 예측
    predicted_class = waste_types[np.argmax(prediction)]  # 예측된 클래스 이름

    plt.imshow(image)  # 이미지 표시
    plt.title(f"Predicted: {predicted_class}")  # 예측 결과 표시
    plt.axis('off')  # 축 숨김
    plt.show()

# 테스트 데이터에서 무작위 이미지로 예측 실행
test_images = test_generator.filenames
for i in range(2000):  # 5개의 이미지를 시각화
    test_image_path = os.path.join(test_generator.directory, test_images[i])
    predict_and_visualize(test_image_path)

```

- `os, random, shutil, pathlib`: 파일 처리 및 디렉토리 생성, 데이터 분할에 사용
- `numpy`: 배열 및 수치 연산 처리
- `tensorflow`: 딥러닝 모델 생성 및 학습
- `ImageDataGenerator`: 이미지 데이터를 전처리 및 증강
- `sklearn`: 평가 지표 계산(혼동 행렬, 분류 보고서
- `matplotlib, seaborn`: 시각화 도구
- 데이터셋 경로와 학습 설정 값을 변수로 지정
- 나중에 코드 전반에서 쉽게 사용 가능
- `train`, `valid`, `test`라는 3개의 데이터셋으로 나누고, 각 데이터셋 하위에 6가지 쓰레기 유형 디렉토리 생성
- `split_indices` 함수는 파일을 무작위로 섞은 후, **훈련(train)**, **검증(valid)**, 테스트(test)로 분할
- 분할 비율은 기본적으로 `50:25:25`
- 각 쓰레기 유형별로 데이터를 훈련, 검증, 테스트 디렉토리에 분류하여 복사
- 이미지 데이터를 `ImageDataGenerator`를 통해 로드하고, 0~1 사이로 정규화
- `train_generator`: 훈련 데이터
- `valid_generator`: 검증 데이터
- CNN(합성곱 신경망)을 사용해 교사 모델 생성
- 데이터의 정답 라벨을 이용해 교사 모델 학습
- 학습 후 모델 저장
- 경량화된(층을 더 얇게 한) CNN 모델로 학생 모델 생성
- 교사 모델의 출력(soft targets)과 학생 모델의 출력 간 차이를 최소화하는 지식 증류 손실 함수 정의
- 교사 모델의 출력(`teacher_logits`)을 학생 모델이 학습하도록 함
- `GradientTape`를 사용해 역전파로 학습
- 테스트 데이터로 모델의 성능 평가
- 혼동 행렬 및 분류 보고서 출력
- 임의의 테스트 이미지를 예측하고, 결과 시각화

</details>




- Confusion Maxtrix
<img width="909" alt="스크린샷 2025-01-05 오후 8 35 00" src="https://github.com/user-attachments/assets/dc2eab47-08a1-4906-9f5d-3a1f75ff0ac2" />

- `cardboard`는 74개 샘플이 정확히 분류
- `cardboard` 중 7개가 `glass`로, 8개가 `metal`로 잘못 분류
- `paper` 클래스에서 가장 높은 정확도
- `cardboard`와 `glass` 간의 혼동이 비교적 높게 나타남
- 확실히 teacher 모델보다 혼동이 빈번하게 발생



### 실제 테스트 
<details>
<summary>실제 테스트 결과 (성공사례)</summary>
<br>

<img width="638" alt="스크린샷 2025-01-08 오후 9 28 37" src="https://github.com/user-attachments/assets/5a4a44f5-6128-42d4-8e81-6d67cfc6e9a3" />
<img width="637" alt="스크린샷 2025-01-08 오후 9 28 21" src="https://github.com/user-attachments/assets/11f813d2-bdaf-4967-baf2-158e4b932f49" />
<img width="637" alt="스크린샷 2025-01-08 오후 9 26 58" src="https://github.com/user-attachments/assets/ecc97732-f89b-4457-8b22-c8215a944d3f" />
<img width="635" alt="스크린샷 2025-01-08 오후 9 26 48" src="https://github.com/user-attachments/assets/28d6487e-cfda-4dbf-a18b-bdcf70a73be2" />
<img width="635" alt="스크린샷 2025-01-08 오후 9 25 53" src="https://github.com/user-attachments/assets/d7773104-1c3e-43f7-81d7-7f15783c3ef6" />
<img width="636" alt="스크린샷 2025-01-08 오후 9 25 45" src="https://github.com/user-attachments/assets/bfa0eb2b-2e3b-43f8-8ecb-6667615c9ce5" />
<img width="635" alt="스크린샷 2025-01-08 오후 9 25 35" src="https://github.com/user-attachments/assets/cf1f6939-d100-439b-b8ef-a96ccfb507f1" />
<img width="637" alt="스크린샷 2025-01-08 오후 9 25 00" src="https://github.com/user-attachments/assets/97719a58-b361-4566-8c0b-51a8dbb7fbac" />
<img width="637" alt="스크린샷 2025-01-08 오후 9 24 44" src="https://github.com/user-attachments/assets/c0a02421-2f8c-4d97-a580-59b9c0922a7b" />


</details>


# 라즈베리파이에서 성능 비교 실험

라즈베리파이에서 지식 증류로 생성한 teacher모델과 student모델의 성능 비교 실험

<details>
<summary>성능 측정 코드 on Raspberry Pi</summary>
<br>

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import cv2
import time
from sklearn.metrics import confusion_matrix, classification_report

# Qt 플러그인 문제 해결
os.environ["QT_QPA_PLATFORM"] = "offscreen"  

# 라즈베리파이에서 모델 성능 평가 코드
# 1. 경로 및 파라미터 설정
MODEL_PATH_TEACHER = "waste_teacher_model.h5"  # Teacher 모델 경로
MODEL_PATH_STUDENT = "waste_student_model.keras"  # Student 모델 경로
TEST_DIR = "data/test"  # 테스트 데이터 디렉토리
IMG_SIZE = (128, 128)  # 이미지 크기
BATCH_SIZE = 32  # 배치 크기

# 2. 테스트 데이터 로더 설정
# 이미지를 불러오고 전처리하는 ImageDataGenerator 생성
# 라즈베리파이에서 메모리 문제를 피하기 위해 배치 크기를 유의미하게 설정

data_gen = ImageDataGenerator(rescale=1./255)

# 테스트 데이터 생성기
test_generator = data_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=1,  # 한 번에 하나씩 로드 (라즈베리파이 메모리 절약)
    class_mode='categorical',
    shuffle=False
)

# 실제 클래스 이름 목록 가져오기
waste_types = list(test_generator.class_indices.keys())

# 3. Teacher와 Student 모델 불러오기
# Teacher 모델 로드
print("Loading teacher model...")
teacher_model = tf.keras.models.load_model(MODEL_PATH_TEACHER)

# Student 모델 로드
print("Loading student model...")
student_model = tf.keras.models.load_model(MODEL_PATH_STUDENT)

# 4. 모델 성능 평가
# Teacher 모델 평가
def evaluate_model(model, generator, model_name):
    print(f"Evaluating {model_name} model...")
    results = model.evaluate(generator, verbose=1)
    print(f"{model_name} Model Accuracy: {results[1]:.4f}")
    return results

evaluate_model(teacher_model, test_generator, "Teacher")
evaluate_model(student_model, test_generator, "Student")

# 5. Confusion Matrix 및 Classification Report 생성
# 테스트 데이터에 대한 예측 생성
print("Generating predictions for Student model...")
y_pred = student_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)  # 예측된 클래스 인덱스

# 실제 클래스 레이블 가져오기
y_true = test_generator.classes

# Classification Report 출력
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=waste_types))

# 6. 테스트 이미지 시각화 및 예측 (OpenCV 사용)
def predict_and_visualize(image_path, model, model_name):
    print(f"Visualizing predictions for {model_name} model...")
    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = waste_types[np.argmax(prediction)]

    # OpenCV로 이미지 읽기 및 표시
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (400, 400))  # OpenCV에서 표시할 크기로 리사이즈
    cv2.putText(img_cv, f"{model_name} Prediction: {predicted_class}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(f"{model_name} Prediction", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 테스트 데이터셋에서 임의의 이미지를 선택하여 예측
test_images = test_generator.filenames
for i in range(5):  # 5개의 이미지를 테스트
    test_image_path = os.path.join(test_generator.directory, test_images[i])
    predict_and_visualize(test_image_path, student_model, "Student")

# 7. 모델 추론 속도 측정
# 추론 속도 측정 함수
def measure_inference_speed(model, generator, model_name, num_samples=100):
    print(f"Measuring inference speed for {model_name} model...")
    start_time = time.time()

    for i, (x_batch, _) in enumerate(generator):
        if i >= num_samples:
            break
        model.predict(x_batch)  # 예측 수행

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_samples
    print(f"{model_name} Average Inference Time per Sample: {avg_time_per_sample:.6f} seconds")
    return avg_time_per_sample

# Teacher와 Student 모델의 추론 속도 측정
measure_inference_speed(teacher_model, test_generator, "Teacher", num_samples=100)
measure_inference_speed(student_model, test_generator, "Student", num_samples=100)

```


</details>


# 최종 결과



https://github.com/user-attachments/assets/00d2e000-2975-4b52-b9e8-d008ee0aa65d


