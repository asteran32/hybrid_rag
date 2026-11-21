# Phase 3 출처 기반 답변 검증

2026년 11월 21일, 구현 커밋 `0f0a542`를 검증했다. 기존 EnterpriseRAG-Bench 표본을 대상으로 실행한 요청 7건 중 4건이 인용을 포함한 부분 답변을 반환했고 3건은 보류됐다. 반환된 인용 4건을 PostgreSQL 원문, 청크, 벤치마크 원본 JSON과 각각 대조했다. 서로 다른 인용 구간은 2곳이었다.

## 실행 조건

| 항목 | 설정 |
| --- | --- |
| 데이터 | EnterpriseRAG-Bench `d36685e273713975ee20299bbf1ab64165575b3c`, 9개 소스의 문서 180개, 청크와 임베딩 각 1,248개 |
| 허용 범위 | Linear 20개 문서·78개 청크, Gmail 20개 문서·152개 청크. 개별 요청의 소스 필터를 추가 적용 |
| 저장소·검색 | Python 3.11.13, PostgreSQL 17.11, pgvector 0.8.2. 키워드는 영어 FTS, 벡터는 로컬 Model2Vec, 하이브리드는 후보 50개·RRF 상수 60 |
| 생성 | Apple M3, llama.cpp `b7712`, Qwen2.5-1.5B-Instruct Q4_K_M, 온도 0, 프롬프트 `grounded-answer-v4`, 출력 한도 1,200토큰 |
| 근거 검증 | mDeBERTa-v3-base-mnli-xnli `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`, 양자화 ONNX, CPU. 지지 확률이 0.50을 초과할 때 채택하고 모순 확률 0.90 이상을 반대 근거로 처리 |

생성 가중치와 서버 바이너리의 SHA-256을 배포본과 대조했다. 생성 가중치의 SHA-256은 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`이다. 질문은 표본 원문을 보고 작성했다. 벤치마크 정답 파일은 런타임 입력에 포함하지 않았다. 유료 API 호출은 0건이다. [생성 모델](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9) · [서버 릴리스](https://github.com/ggml-org/llama.cpp/releases/tag/b7712) · [근거 검증 모델](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)

## 테스트와 검증 방법

| 검증 | 확인한 내용 | 결과 |
| --- | --- | --- |
| 단위 테스트 | 기존 수집·검색·지표, 유니코드와 반복 문장의 위치, 가짜 인용 거부, 사실·추론, 수치·약어, 주장 지지와 충돌, 생성 거부와 출력 한도 | 46개 통과 |
| PostgreSQL 통합 테스트 | 기존 저장·검색, 순위 계산 전 권한 적용, 생성 중 삭제·수정·권한 변경, 세 검색 방식의 CLI, FastAPI 응답과 오류 | 18개 통과 |
| 전체 테스트 재실행 | 프로젝트 환경에서 64개를 실행하고, 설치용 wheel을 별도 Python 환경에 설치한 뒤 저장소 밖에서 같은 테스트를 재실행 | 각각 64개 통과, 실패·건너뜀 0개. 1.29초와 1.45초 |
| 패키지와 정적 검사 | 설치한 wheel에서 모듈·SQL 스키마·HTTP 경로 확인, `uv lock --check`, Ruff 검사와 포맷 검사 | 모두 통과. 포맷 검사 대상 45개 파일 |
| 실제 HTTP 서버 | 공백 질문, 요청 본문의 권한 확장, `gold_answer` 주입을 요청 | 3건 모두 HTTP 422 |
| 실제 모델 출력 한도 | 검증 프로세스의 한도를 8토큰으로 낮추고 출력을 요구 | 8토큰에서 종료, `output_limit` |

인용 대조 스크립트는 답변 서비스의 검증 함수를 호출하지 않았다. 반환된 `document_id`와 `chunk_id`로 직접 조인하고, 두 테이블에서 각각 문자열을 잘라 `quote`와 비교했다. 이어 `source_path`로 원본 JSON을 열어 `dataset_doc_uuid`를 확인하고, 원본이 지정한 본문 필드의 순서로 텍스트를 재구성해 같은 위치를 비교했다. 모든 위치는 0부터 세며 끝 위치는 제외한다.

## EnterpriseRAG-Bench 표본 결과

| 질문·진입점 | 검색 | 결과 | 인용 수 | 소요 시간 |
| --- | --- | --- | ---: | ---: |
| 교대 근무자용 서랍이 표시되는 두 페이지 · CLI | 키워드 | `partial` · Alerts와 SLO | 1 | 7.960초 |
| 같은 질문 · API | 벡터 | `partial` · Alerts와 SLO | 1 | 4.569초 |
| 같은 질문 · API | 하이브리드 | `partial` · Alerts와 SLO | 1 | 5.247초 |
| 사용성 테스트에 몇 명이 참여하는가 · API | 키워드 | `abstained` · 제시한 인용에서 인원수를 확인하지 못함 | 0 | 0.899초 |
| InnovaCare가 요청한 단가 동결 기간 · API | 키워드 | `partial` · 3년 | 1 | 3.281초 |
| 일치하는 어휘가 없는 질문 · API | 키워드 | `abstained` · 생성 호출 0회 | 0 | 0.018초 |
| 허용 범위 밖인 Slack으로 제한한 질문 · API | 하이브리드 | `abstained` · 생성 호출 0회 | 0 | 0.003초 |

세 검색 방식이 인용한 Linear 문서는 `dsid_1d5c022db7ef4708a7178bc291921770`, 구간은 `[166, 673)`이다. Gmail 답변은 `dsid_02892bbb04c94f5bbeaf5f0068ba47d8`의 `[301, 495)`에서 3년이라는 기간을 확인했다. 표의 시간은 각 요청 1회의 전체 경과 시간이며 CLI에는 프로세스 시작 시간도 포함된다.

## 통제된 문서로 확인한 실패 처리

| 입력 근거와 질문 | 실제 로컬 모델을 거친 결과 |
| --- | --- |
| 보관 기간이 30일이라고 적힌 문서에 기간만 질문 | `partial` · 30일을 인용 |
| 같은 문서에 보관 기간과 연간 갱신 비용을 함께 질문 | `partial` · 30일만 반환 |
| 같은 문서에 갱신 비용만 질문 | `abstained` · 인용으로 뒷받침할 수 없는 금액을 제거 |
| 회의실 정원을 두 공지가 각각 12명, 20명으로 명시 | `abstained` · 두 공지의 원문을 `conflicts`에 보존 |
| 영문 문서에 명시된 메뉴 담당자 질문 | `partial` · Maya Chen을 인용 |
| 한국어 문서에 명시된 자료실 담당자 질문 | `abstained` · 생성된 주장에 원문에서 확인할 수 없는 약어가 들어감 |

초기 실행에서는 모델이 보관 기간을 인용하고 갱신 비용을 100달러라고 답했다. 최종 구현은 발췌문에 없는 숫자·영문 대문자 약어를 검사하고, 별도 NLI 모델로 주장과 인용의 관계를 확인한다. 같은 주제·단위·조건의 명시적 수치가 다른 경우에는 해당 원문 두 곳을 보존한다. 위 두 오류를 회귀 테스트에 포함했다.

## 재현과 해석

README에 따라 표본 DB와 로컬 모델 서버를 준비하고 `.env`에 접속 정보와 허용 범위를 지정한다. 통합 테스트에는 임시 스키마를 만들 수 있는 `TEST_DATABASE_URL`이 필요하다. [설정과 답변 실행 방법](../README.md#4-출처를-포함한-답변)

```bash
uv sync --locked --all-extras
uv run --locked --all-extras pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests

for method in keyword dense hybrid; do
  uv run --locked --extra answer enterprise-agent answer \
    'On which two Redwood Console pages is the shift-change companion drawer available?' \
    --method "$method" --source-type linear --top-k 3
done
```

질문별 원본 응답, 입력 설정과 실행 스크립트는 이 체크아웃의 로컬 결과 디렉터리에 보관했다. [표본 요청과 측정값](results/phase3-checked-20261121T031054Z-tCZIBp/sample-summary.json) · [원문 대조 스크립트](results/phase3-checked-20261121T031054Z-tCZIBp/sample_checks.py) · [통제 문서 실행 스크립트](results/phase3-checked-20261121T031054Z-tCZIBp/fixture_checks.py) · [테스트 원본 XML](results/phase3-checked-20261121T031054Z-tCZIBp/tests.xml)

실제 표본에서 모델은 질문에 맞는 사실을 반환하면서도 답변 상태를 보수적으로 `partial`로 표시했다. 사용성 테스트 문서에는 실제로 5명이 명시돼 있지만, 모델이 선택한 구간에는 그 수가 없어 보류했다. `answered` 경로는 통제된 생성 응답을 사용한 CLI·HTTP 통합 테스트에서 검증했다. 이 실행은 Phase 3의 출처 추적과 실패 처리를 확인한 표본 검증이다. 공식 500문항의 답변 정확도·완전성은 별도 평가 대상이며, 권한 검증의 범위는 인스턴스에 설정한 허용 목록까지다.
