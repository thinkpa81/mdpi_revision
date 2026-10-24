# JTAER 최종 논문 분석자료

이 저장소는 **Questionnaire Item Budgets and Incremental Predictive Value for Self-Reported AI-Enabled Consumer Service Use: Temporal Evidence from a Korean Panel**의 분석 코드, 집계 결과, 그림과 출처 기록을 제공합니다. 최신 수정에서는 제출 문서 7개의 후향적·탐색적 분석 범위, 측정 진단, 후속 검증 설계 및 재현성 안내를 정리했습니다. 답변서에 인용된 편집자·심사자 의견 34개는 유지했습니다. 상세 위치 안내 105개를 수정 문서와 대조하여 페이지·줄 위치 23개를 갱신했습니다. 일반본과 붉은색 표시본의 본문·수식 및 페이지 수가 일치함을 확인했습니다. 최신 문서 식별 정보와 검증 범위는 [EDITORIAL_ALIGNMENT_CHECK_20261024.json](EDITORIAL_ALIGNMENT_CHECK_20261024.json)에 기록했습니다. 이전 답변서 검토는 [RESPONSE_ALIGNMENT_CHECK_20261024.json](RESPONSE_ALIGNMENT_CHECK_20261024.json), 이전 제목·배포 정리는 [FINAL_ALIGNMENT_CHECK.json](FINAL_ALIGNMENT_CHECK.json)에 남아 있습니다. 제출 문서 자체는 공개 저장소에 포함하지 않습니다.

## 자료 구성

- `main_result/revision/code/`: 현재 분석·검증·그림 생성 코드
- `main_result/revision/tables/`, `main_result/revision/results/`: 모형 결과, 부트스트랩 집계, 선택·중복 참여·지역·인접 서비스 진단
- `main_result/revision/audit/`: 소스 해시, 분석 계획, 소프트웨어 환경과 실행 기록
- `main_result/original_results/`: 표본 구성, 원래 결과와의 비교, 유지된 nested 교차검증 및 보충표 S25에 필요한 이전 분석자료
- `JTAER_Figures/`: 본문 그림 1–7, 보충 그림 S1–S4의 PNG·PDF·SVG 및 집계 원천 CSV
- `SOURCE_RECHECK_20261024.json`: 보충자료 S10이 인용하는 원자료 재검증 기록

보충표 S26의 코드·결과 경로를 유지했습니다. 초기 부트스트랩 200회 기록은 최종 1,000회 분석의 앞부분과 일치하는지 검증하는 데 사용하므로 보존했습니다. 초기 200회는 별도의 독립 표본이 아닙니다.

## 최신 문서 수정

후속 기간의 결과를 이미 검토한 후향적·탐색적 이차분석이라는 점과 추가 튜닝을 개발 데이터에 한정한다는 점을 명시했습니다. 평가 코호트의 재사용은 추가 탐색 평가로 설명하고, 원래의 비교군과 추가 민감도 분석을 구분했습니다. 표 S28은 실제 확보한 근거와 해석·검증 범위를 구분하며, 제안된 동시기 문구 비교 연구가 아직 수행되지 않았음을 명시합니다. 점별 구간과 동시 구간의 범위, 중첩 반복의 의존성, 측정동등성 및 인과효과의 식별 한계는 유지했습니다. 실제 수집 정책의 전향적 평가에는 안정적인 측정과 직접 측정한 획득 비용이 필요합니다.

본문의 반복적인 저장소·그림 경로 안내는 Data Availability Statement와 보충자료 S10·S26에 모았습니다. 공개 그림 11개의 캡션과 Word 삽입 PNG를 대조하고 문서 식별 정보를 갱신했습니다. 이번 단계에서 새 모형 적합이나 집계 재계산은 수행하지 않았습니다.

## 이전 배포 정리

현재 실행에 사용되지 않는 과거 코드 사본 8개, 중복 캡션 문서 1개, 이전 업로드 준비 점검 기록 1개를 현재 파일 목록에서 정리했습니다. 삭제한 파일의 해시와 이전 커밋 주소를 기록하여 출처 확인이 가능합니다. 현재 분석 코드, 집계 수치, 그림 원천 데이터와 이미지 파일은 유지했습니다. 그림 캡션과 문서 식별 정보를 최종본에 맞추고, 과거 버전명·업로드 전 상태를 설명하던 안내를 갱신했습니다.

개인별 원자료·파생자료·식별자·예측값·교차검증 배정·모형 객체는 공개 범위에서 제외합니다. 논문·보충자료 Word/PDF, 답변서, 이메일, 원본 압축파일과 KISDI 원본 조사표·코드북도 업로드하지 않습니다. 제공된 PDF 파일은 그림의 벡터 출력본입니다.

## 공개 자료 검증

```bash
python verify_release.py
python -m pip install -r main_result/revision/requirements.txt
cd main_result
python validate_public_release.py
python run_reproduction.py --dry-run
```

첫 검사는 파일 무결성·Python 문법·JSON을 확인합니다. 집계 검사는 8개 조건별 1,000회, 총 8,000개 반복별 집계 레코드에서 구간과 선택 빈도를 재구성합니다. 모델을 새로 적합하는 검사가 아닙니다. 앞선 답변서 검토에서는 공개 근거 파일 98개의 해시를 기준 커밋의 manifest와 대조하고 8,000개 기록을 재집계했습니다. 재계산의 최대 절대 차이는 9.94e-17 미만이었습니다. 전체 저장소 검증기나 원자료 분석을 새로 실행한 것은 아니며, 과학적 코드·수치·그림은 변경하지 않았습니다. 재실행을 원하면 KISDI 이용 조건에 따라 원자료를 별도로 확보한 후 `main_result/README.md`를 따릅니다. 보충표 S26의 실행 진입점은 `main_result/run_reproduction.py`입니다.

그림 재생성 경로는 `main_result/JTAER_Figures/`이며, 저장소 최상위 `JTAER_Figures/`가 최종 논문 번호를 적용한 배포본입니다. 두 경로의 대응과 최종 캡션은 `FIGURE_CROSSWALK.csv`에 있습니다. 11개 PNG는 최종 Word 문서에 삽입된 이미지와 바이트 단위로 일치합니다.

## 원자료 재검증 범위

별도 원자료 재검증에서 원자료·코드북·조사표 9개 파일의 해시를 확인하고, 표본 재구성 및 조사표·코드북·인접 서비스 진단 코드를 실행했습니다. 분석 표본 4개의 해시와 집계 CSV 20개가 기존 결과와 일치했습니다. 인접 서비스 진단은 코호트별 참여자 부트스트랩 10,000회, 대조 36개, 조건부 검정 24개를 다시 계산했습니다. 예측 모형 전체 적합이나 개발 절차 부트스트랩 1,000회를 다시 실행한 것은 아닙니다. 세부 기록은 [SOURCE_RECHECK_20261024.json](SOURCE_RECHECK_20261024.json)에 있습니다.

## 인용과 이전 결과

최종 문서가 인용하는 [`af06ecda65c0f5768cbe2b3449bb46ebf229edca`](https://github.com/thinkpa81/mdpi_revision/commit/af06ecda65c0f5768cbe2b3449bb46ebf229edca) 커밋은 그대로 유지됩니다. 현재 수정은 이 커밋의 과학적 결과를 유지하면서 문서 식별 정보와 해석 범위 안내를 맞춘 것입니다. 삭제된 과거 사본은 해당 커밋에서 확인할 수 있습니다. 기존 [thinkpa81/mdpi](https://github.com/thinkpa81/mdpi) 저장소는 수정하지 않았습니다. 특정 버전 인용 시 사용한 실제 GitHub 커밋을 명시하십시오.

