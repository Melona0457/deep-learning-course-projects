# Deep Learning Course Projects

> CNN, Transformer, Vision Transformer, BERT fine-tuning 실습을 공개용 스크립트로 정리한 딥러닝 과제 저장소입니다.

## Portfolio Focus

이 저장소는 딥러닝 모델을 단순히 사용하는 데서 끝나지 않고, **데이터 로딩부터 모델 구조, 학습 루프, 평가 흐름까지 직접 다뤄본 경험**을 보여주는 데 초점을 둡니다. 원본 노트북은 공개에 맞지 않는 출력과 개인 경로가 포함되어 있어, 실행 코드만 추출해 스크립트 형태로 재구성했습니다.

## Project Index

| Script | Topic | Summary |
| --- | --- | --- |
| `scripts/assignment1_cnn.py` | CNN image classification | CIFAR-10 기반 CNN, residual/inception block, global average pooling 실습 |
| `scripts/assignment2_transformer_from_scratch.py` | Transformer | attention, encoder block, 학습 루프 구현 |
| `scripts/assignment2_vit.py` | Vision Transformer | patch embedding과 transformer encoder 기반 이미지 분류 |
| `scripts/final_project_bert_finetuning.py` | NLP fine-tuning | BERT 기반 텍스트 분류 fine-tuning 흐름 정리 |

## Technical Highlights

- PyTorch 기반 `Dataset`, `DataLoader`, model, optimizer, loss, validation loop를 직접 구성했습니다.
- CNN 과제에서는 구조 변경을 통해 feature extraction 성능을 개선하는 실험을 진행했습니다.
- Transformer 계열 과제에서는 입력 embedding, attention, encoder block, classifier head의 데이터 흐름을 코드로 확인할 수 있습니다.
- BERT fine-tuning에서는 사전학습 모델을 downstream classification task에 맞게 조정하는 흐름을 정리했습니다.

## Repository Structure

```text
env/
  environment.yml
scripts/
  assignment1_cnn.py
  assignment2_transformer_from_scratch.py
  assignment2_vit.py
  final_project_bert_finetuning.py
```

## How to Run

```bash
conda env create -f env/environment.yml
conda activate ai-assignment
python scripts/assignment1_cnn.py
```

스크립트는 수업 실습 환경에서 작성된 코드라 로컬 실행 시 데이터 경로, GPU 사용 여부, 패키지 버전을 환경에 맞게 조정해야 할 수 있습니다.

## Public Release Notes

- 원본 `.ipynb`의 출력, Colab 메타데이터, 개인 Google Drive 경로는 제거했습니다.
- 학습된 `.pth` 모델과 대용량 데이터셋은 포함하지 않았습니다.
- 보고서와 수업 명세 대신 구현 코드 중심으로 공개했습니다.

## Retrospective

모델 성능은 구조 자체뿐 아니라 데이터 전처리, batch size, optimizer, epoch 설정의 영향을 크게 받았습니다. 특히 Transformer 계열 코드를 직접 따라가며 attention이 입력 tensor shape에 얼마나 민감한지 체감했고, 실험 코드를 공개 가능한 형태로 정리하면서 재현 가능한 프로젝트 구조의 중요성도 함께 배웠습니다.
