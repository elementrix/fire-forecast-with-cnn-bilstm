# fire-forecast-with-cnn-bilstm

This repository contains experiments for wildfire forecasting using CNN-BiLSTM
architectures as well as utilities that support related coursework.  The code
base has been augmented with a constraint-programming solution for the "신학기
학생배정" assignment.

## Structure

| File | Description |
| ---- | ----------- |
| add_pos_weight_cnn_bilstm.py | main CNN-BiLSTM forecasting script |
| checking_data_set.py | data preparation script |
| cnn-bilstm-base.py | alternative baseline model |
| get_from_ftp/ | utilities to fetch remote datasets |
| student_assignment.py | OR-Tools based solver for the 학생배정 homework |

## 신학기 학생배정 과제 도우미

`student_assignment.py`는 Google OR-Tools CP-SAT을 사용하여 1학년 200명의
학생을 6개의 학급으로 배정하는 문제를 해결합니다. 엑셀로 제공될 학생
데이터를 읽어 다음 제약조건을 만족하도록 학급을 배정합니다.

- 문제 아동끼리는 같은 학급에 배정하지 않음
- 챙겨주는 친구는 반드시 같은 학급으로 배정
- 각 학급에 리더십이 있는 학생이 최소 1명 이상
- 피아노 가능자, 운동 능력자, 비등교 위험 학생을 균등 분배
- 남녀 비율 균형 유지
- 전년도 같은 반이었던 학생과 동아리 활동 인원을 골고루 분배
- 성적 분포(사분위수 기준)를 균등하게 유지

### 사용법

```bash
python student_assignment.py dataset.xlsx --output assignments.csv --verbose
```

필수 컬럼은 `id`, `score`이며, 나머지 제약 조건에 필요한 컬럼 이름과 형식은
`student_assignment.py` 상단의 문서를 참고하세요. 기본 학급 정원은
`33,33,33,33,34,34`로 설정되어 있으며 `--class-sizes` 옵션으로 변경할 수
있습니다. 실행이 성공하면 학급 배정 결과가 CSV로 저장되며, `--verbose`
옵션을 사용하면 각 학급에 대한 요약 통계를 출력합니다.

필요한 파이썬 패키지는 `pandas`, `ortools`, `openpyxl`이며 아래와 같이
설치할 수 있습니다.

```bash
pip install pandas ortools openpyxl
```
