# 설문 문항 예산과 AI 서비스 이용의 시간순 예측

장순호·서응교, 단국대학교.

현재 재현 자료는 이 저장소에서 통합하여 관리합니다. 버전 식별자는 `v2.0-minor-revision`이며, GitHub 릴리즈에 해당 전체 스냅샷의 커밋이 기록됩니다. [릴리즈 정보](reproducibility/RELEASE.json), [변경 내역](RELEASE_NOTES.md), [기능별 파일 안내](reproducibility/INDEX.json)를 참조하십시오.

기존 분석 폴더와 실행 경로는 코드·출처 링크의 연속성을 위해 유지하였습니다. 날짜가 붙은 폴더는 역사적 분석 묶음을 식별하며, 전체 저장소의 버전은 커밋과 릴리즈 태그로 구분합니다. 이번 공개 자료 정리는 모형 재적합이나 분석 코드·집계 결과·그림 데이터의 변경을 포함하지 않습니다.

이전 `thinkpa81/mdpi` 저장소는 역사적 보관본입니다. 최초 공개 코드의 커밋은 `f3b0a46b86f8ab0a7e6093f17105d0f38d281824`이며, `research_release_20261021` 묶음은 `34f4172e3bebdb97fe5d5044dd51b237a7c82881`에 있습니다. 서로 다른 두 시점을 하나의 초기 제출 태그로 표현하지 않았습니다. 이후 변경은 현재 저장소에서 관리합니다.

공개 범위는 코드·집계 결과·그림·출처 기록입니다. 개인 수준 원자료는 KISDI의 접근 및 통계 이용 조건을 따릅니다. 원고, 심사 대응 문서, 참여자 기록·식별자·개인별 예측값은 공개 범위에 포함되지 않습니다.

파일 무결성과 구문은 `python verify_release.py`로 확인할 수 있습니다. 주요 실행 진입점은 `main_result/run_reproduction.py`이며, 생성형 AI·중복 진단과 소프트웨어 검증·시드 목록·갱신된 그림 7은 `minor_revision_20261101_v1/`에 있습니다.

## 세부 재현 절차와 분석 출처

## 주요 경로

| 경로 | 내용 |
| --- | --- |
| [minor_revision_20261101_v1](minor_revision_20261101_v1/README.md) | 최신 진단 코드·집계 결과·재현 확인 자료 |
| [시드 목록](minor_revision_20261101_v1/qa/public_seed_catalog.json) | 분석군별 시드, 적용 범위 및 집계 결과 확인 |
| [Figure 7](minor_revision_20261101_v1/figures) | 새 패널 표시를 적용한 그림, 생성 코드, 집계 데이터 및 시각 비교 기록 |
| [main_result/revision](main_result/revision) | 기존 복원·모형·추론·민감도 분석 코드와 결과 |
| [JTAER_Figures](JTAER_Figures) | 이전 그림과 집계 데이터; Figure 7의 최신 표시 버전은 위 추가 자료에 수록 |
| [FILE_MANIFEST_SHA256.json](FILE_MANIFEST_SHA256.json) | 자기 자신을 제외한 공개 파일의 SHA-256 |

날짜가 붙은 기존 문서 정합성 확인서는 해당 날짜의 문서에 관한 기록입니다. 이후 문서의 검증 결과로 해석하지 않습니다. 논문 Word 파일, 답변서 및 심사 의견은 이 공개 저장소에 포함하지 않습니다.

## 원자료 없이 확인하기

```bash
python verify_release.py
python -m pip install -r minor_revision_20261101_v1/requirements.txt
cd minor_revision_20261101_v1
python qa/verify_public_addendum.py
```

첫 명령은 전체 공개 파일의 해시와 Python·JSON 구문을 확인합니다. 추가 자료 검증기는 파일 유형, CSV 구조 및 부트스트랩 집계값의 정확한 일치를 확인합니다. 이 검증은 새 모형을 적합하지 않습니다.

초기 200회 부트스트랩을 원래 보존된 적합 코드와 시드, 마지막 800회에 기록된 소프트웨어 버전으로 다시 생성했습니다. 집계 1,600행과 점추정치 8행이 정확히 일치했으며, 재생성한 200회와 기존 800회를 합친 구간 및 선택 빈도도 변하지 않았습니다. 최초 200회 기록에는 별도의 버전 확인서가 없으므로, 과거 환경의 동일성을 추정하지 않고 재생성 결과를 근거로 제시합니다.

원래 분석의 집계 검증기는 저장소 루트에서 `cd main_result` 후 `python validate_public_release.py`로 실행할 수 있습니다. 전체 파이프라인 순서 점검은 `python run_reproduction.py --dry-run`입니다.

## 원자료를 이용한 재현

2023–2025년 KISDI AI 서비스 이용자 정책 아카이브 자료는 제공기관의 이용 조건에 따라 별도로 확보해야 합니다. 기존 파이프라인은 `main_result/private_raw/`에 `YYYY_data.xlsx`, `YYYY_codebook.xlsx`, `YYYY_questionnaire.pdf`를 배치한 뒤 [main_result/README.md](main_result/README.md)를 따릅니다. 추가 진단의 별도 파일명과 실행 명령은 [추가 자료 안내](minor_revision_20261101_v1/README.md)에 있습니다. 실행 시 결과와 기록이 덮어써질 수 있으므로 별도 작업 사본을 사용하십시오.

시드 20261020은 원래 고정 예측값 비교군에 적용됩니다. Figure 2의 절대 로그손실 점별 구간은 2026102002를 사용합니다. 기여도·지역 재코딩 등 다른 분석의 시드는 별도 목록에 명시했습니다. 이번 설명 정리로 시드나 수치 결과를 바꾸지 않았습니다. Figure 7은 패널 문자만 변경했으며 값·축·색상·유의성 테두리는 유지했습니다.

## 공개 범위와 인용

공개 대상은 코드, 메타데이터, 집계 통계 및 그림입니다. 개인별 응답, 식별자, 예측값, 재표집 빈도, 폴드 배정, 적합 객체, 비공개 캐시, 원자료·설문지 전체, 투고 문서 및 심사 서신은 제외합니다.

인용할 때는 실제 사용한 파일이 포함된 Git 커밋을 지정하십시오. 새 커밋의 해시를 해당 커밋 파일 안에 넣으면 자기참조가 생기므로 이 안내에는 새 해시를 기재하지 않습니다. 이전 수치 분석 스냅샷은 [`af06ecda65c0f5768cbe2b3449bb46ebf229edca`](https://github.com/thinkpa81/mdpi_revision/commit/af06ecda65c0f5768cbe2b3449bb46ebf229edca)에 보존되어 있습니다. 이번 추가 자료는 소프트웨어 라이선스나 제공기관 원자료의 재배포 권한을 새로 부여하지 않습니다.
