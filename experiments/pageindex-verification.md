# PageIndex 검색 구현 검증

2026년 11월 22일, `pageindex` 브랜치의 `e907ed4`에서 EnterpriseRAG-Bench의 9개 소스별로 20문서씩 색인했다. 검증 범위는 이 180문서에서의 검색과 근거 인용이다.

## 구현과 색인

| 단계 | 적용한 내용 | 커밋 |
| --- | --- | --- |
| 텍스트를 목차로 변환 | 소스 필드·기존 제목·회의 발언을 Markdown으로 표현하고 PageIndex가 부여한 노드 ID와 행을 원문 문자 위치에 연결했다. | `e18405b` |
| PostgreSQL에 저장 | 문서 지문과 색인 버전을 확인해 완성된 트리를 재사용하고, 한 문서의 변경 내용을 트랜잭션 하나에서 교체한다. | `6276524` |
| 문서·목차·근거 선택 | 열람 가능한 문서만 모델에 보여 준다. 모델이 선택한 구간이 두 청크에 걸치면 해당 원문을 복원할 수 있는 청크와 범위를 함께 반환한다. | `4d8046b` |
| CLI와 답변 API | 답변기에서 선택된 근거 범위를 우선 읽고 권한·인용 위치·주장의 근거를 재검증한다. | `1a90815` |
| 평가와 로컬 모델 연결 | 후보·최종 문서 순위, 근거 구간, 지연·호출·토큰을 기록한다. 문서 목록은 20개씩 제시하고 유효한 선택지는 재정렬 응답이 잘못된 경우에도 보존한다. | `65880f2`, `3e3bac9`, `eddaa69`, `e907ed4` |

실제 JSON 180개에서 목차 노드 2,026개를 생성했다. 본문이 있는 노드 구간 1,645개를 원문에 대조한 결과, 공백을 제외한 모든 문자가 정확히 한 구간에 속했다. 트리를 담은 PostgreSQL JSONB 열의 크기 합은 311,013바이트다. 같은 DB 연결에서 빈 색인 테이블로 시작한 단일 측정에서 180개 생성에 220.52ms, 동일 입력 재실행에 7.08ms가 걸렸다. 재실행은 180개 모두 재사용했다. 파일과 DB 캐시가 채워진 상태의 값이다.

## 실험 조건

| 항목 | 설정 |
| --- | --- |
| 데이터 | EnterpriseRAG-Bench `d36685e273713975ee20299bbf1ab64165575b3c`. 1,200자 청크, 겹침 160자, 청크 1,248개 |
| 색인 | PageIndex 0.2.10 `add6b17cebff528612748ede3fed565b5f726c6b`의 `md_to_tree()`. 원본 JSON 필드 사용, LLM 요약 꺼짐 |
| 모델 | Apple M3, 통합 메모리 8GB. llama.cpp `b7712`, Qwen2.5-1.5B-Instruct Q4_K_M, 동시 요청 1개, 서버 프롬프트 캐시 한도 640MiB |
| 탐색 | 프롬프트 `pageindex-select-v2.1`, temperature 0, 문맥 8,192토큰 중 출력 384토큰 예약, 후보 문서 최대 20개. 질문당 모델 호출 최대 32회, 형식 오류 재시도 최대 1회, 시간 한도 120초 |
| 비교 | PostgreSQL 영어 전문 검색(C0), 256차원 Model2Vec 벡터 검색(C1), 두 순위를 RRF 상수 60으로 결합한 하이브리드(C2), 로컬 LLM이 문서 목록과 목차에서 선택하는 PageIndex(PI). 각 방식에서 최대 50청크 반환 |

가중치 SHA-256은 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`이다. 입력 예산 계산에는 모델과 일치하는 Qwen 토크나이저 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`을 사용했다. 영문·한글·숫자 등을 포함한 입력 3개의 토큰 ID가 서버와 일치했다. 모델·문서·질문·색인 지문은 실험 설정에 보존한다. [PageIndex Markdown 파서](https://github.com/VectifyAI/PageIndex/blob/add6b17cebff528612748ede3fed565b5f726c6b/pageindex/page_index_md.py) · [Qwen 토크나이저](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

## 재현과 자동 테스트

전용 DB에 문서와 기본 검색 색인을 준비한 뒤, Qwen 모델을 로컬 Chat Completions 서버에 `local-model`로 띄운다. `PAGEINDEX_BASE_URL`에는 서버의 루프백 주소와 `/v1` 경로를 지정한다. 이 측정에서는 서버 옵션 `-c 8192 -np 1 --cache-ram 640`을 사용했다. `PAGEINDEX_TOKENIZER_PATH`에 위 리비전의 `tokenizer.json`을, `PAGEINDEX_MODEL_REVISION`에 가중치 SHA-256을 지정한다. [DB와 모델 준비 방법](../README.md#getting-started)

```bash
uv sync --locked --python 3.11 --all-extras
uv run --locked enterprise-agent build-index --kind pageindex --input EnterpriseRAG-Bench
uv run --locked enterprise-agent evaluate --methods keyword dense hybrid pageindex \
  --questions EnterpriseRAG-Bench/questions.jsonl --limit 500 --chunk-depth 50 \
  --dataset-version d36685e273713975ee20299bbf1ab64165575b3c \
  --output experiments/results/pageindex-sample180.json
```

PostgreSQL에 연결해 `pytest -q`를 실행한 결과 80개 테스트가 통과했다. PageIndex 의존성에서 PyPDF2 사용 중단 경고가 한 번 발생했다. 테스트에는 동시 색인과 중단 후 재시도, 문서 권한과 메타데이터 필터, 잘못된 모델 선택과 호출 한도, 청크를 가로지르는 인용, 벡터 없는 DB에서의 PI 단독 평가가 포함된다. `uv lock --check`, Ruff 검사와 포맷 검사, wheel 빌드도 통과했다. wheel에서 색인·검색 모듈, DB 스키마, 선택 의존성 `pageindex==0.2.10`을 확인했다.
