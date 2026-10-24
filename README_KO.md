# JTAER 수정 논문 분석자료 안내

이 저장소는 논문 **Questionnaire Collection Volume and Predictive Value for Self-Reported AI-Enabled Consumer Service Use: Evidence from a Korean Panel**의 2026년 10월 수정본에 해당하는 공개 분석자료입니다.

지정된 보충자료 폴더의 `main_result.zip`과 `JTAER_Figures.zip`을 해제한 원본 파일 234개를 그대로 수록했습니다. 통계값, 코드, 그림, 기존 실행 기록의 바이트는 변경하지 않았으며, 저장소 최상위에 안내 문서와 검증 파일을 추가했습니다.

## 자료 구성

- `main_result/revision/code/`: 분석·검증·그림 생성 코드
- `main_result/revision/tables/`, `main_result/revision/results/`: 집계 결과와 추가 진단
- `main_result/revision/audit/`: 실행 기록, 소스 해시, 분석 계획과 환경
- `main_result/original_results/`, `main_result/historical_source/`: 현재 결과의 재현과 출처 확인에 필요한 이전 분석자료
- `JTAER_Figures/`: 본문 그림 1–7, 보충 그림 S1–S4의 PNG·PDF·SVG 및 집계 원천 CSV

응답자별 원자료·파생자료·식별자·예측값·교차검증 배정은 포함하지 않습니다. 압축파일 자체, 투고용 Word 파일, 수정 표시본, 심사 대응문, 내부 전달자료는 업로드 범위에서 제외했습니다.

## 공개자료만으로 확인

```bash
python verify_release.py
python -m pip install -r main_result/revision/requirements.txt
cd main_result
python validate_public_release.py
python run_reproduction.py --dry-run
```

첫 번째 검사는 파일 무결성·Python 문법·JSON을 확인합니다. 집계 검사는 8개 조건의 부트스트랩 1,000회씩 총 8,000개 집계 레코드에서 구간과 선택 빈도를 재계산합니다. 여기서 레코드는 개인별 관측치가 아니라 반복별 분석 요약입니다.

업로드 준비 과정에서 두 원본 목록의 해시 232개, Python 46개, JSON 36개를 확인했습니다. 부트스트랩 집계 재구성의 최대 절대 차이는 약 `9.93e-17`이었으며, 실행 순서의 dry run도 통과했습니다. 본문·보충 그림 PNG 11개는 첨부된 2026년 10월 24일 Word 문서 내부 그림과 바이트 단위로 일치했습니다. 이번 업로드 작업에서 원자료를 이용한 모델 적합을 다시 실행하지는 않았습니다.

## 전체 분석 재실행

KISDI의 이용 조건에 따라 원자료를 별도로 확보한 후 `main_result/README.md`를 따릅니다. 재실행은 별도 작업 사본에서 수행하십시오. 새 결과와 기록이 생성되면 배포본 해시와 달라집니다.

보충자료 Table S26에 기재된 이전 실행 스크립트 이름 `run_all_round2.py`에 해당하는 배포용 진입점은 `main_result/run_reproduction.py`입니다. 이 파일의 `--stage` 옵션으로 순서대로 실행할 수 있습니다.

재생성 그림은 `main_result/JTAER_Figures/`에 분석용 이름으로 저장됩니다. 저장소 최상위 `JTAER_Figures/`에는 논문 번호를 적용한 최종 그림이 있으며, 두 이름의 대응은 `FIGURE_CROSSWALK.csv`에 있습니다.

## 버전과 기존 저장소

기존 저장소는 [thinkpa81/mdpi](https://github.com/thinkpa81/mdpi)입니다. 이번 자료는 별도 저장소인 [thinkpa81/mdpi_revision](https://github.com/thinkpa81/mdpi_revision)에 게시하며, 기존 저장소는 수정하지 않았습니다.

`main_result/` 내부 README와 manifest는 원래 압축파일의 출처 기록이므로 과거 버전명, 업로드 전 상태, 미지정 commit 값이 남아 있습니다. 이번 GitHub 게시에 대한 설명은 최상위 `README.md`와 `RELEASE_VALIDATION.json`을 참고하고, 인용 시 실제 GitHub commit을 기록하십시오.
