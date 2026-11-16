# Code Day 7 — Agentic RAG trên vector store Day 6

| File | Việc |
|---|---|
| `config.py` | Đường dẫn Chroma Day 6, embedding/rerank model, model agent, bảng giá token |
| `export_day6.py` | Xuất 599 chunk từ Chroma `day06_strategy_C` → `data/corpus/chunks.json` (kiểm tra vector 768 chiều) |
| `build_benchmark.py` | Chuyển 24 câu Day 6 sang schema Day 7 và thêm 4 câu thiếu evidence (có kiểm tra vắng mặt trên toàn corpus) → `data/benchmark/questions.json` |
| `evidence.py` | Định nghĩa hit (đúng nguồn + trang + chứa trọn evidence sau chuẩn hóa), dùng chung mọi metric |
| `retrievers.py` | 4 phương pháp: `dense` (e5 + Chroma), `bm25`, `hybrid` (RRF k=60), `rerank` (hybrid → cross-encoder) |
| `eval_retrieval.py` | Hit@1, Evidence Hit@3, MRR@10, latency trên câu hỏi gốc |
| `agent.py` | Agent tool calling (OpenAI): `search` / `answer` / `abstain`, context budget, dedup, neighbour, kiểm tra citation, trace |
| `run_agent.py` | Chạy các cấu hình trên dev/test; lưu trace + metric tự động (tool use, CER, token, chi phí, thời gian) |
| `review_sheet.py` | Tạo phiếu chấm tay (câu trả lời + nguyên văn đoạn được cite) và khung `reviews.json` |
| `label_dev.py`, `label_test.py` | Nhãn chấm tay (ghi dưới dạng code để xem lại được) |

## Cài đặt

Python 3.12 (đã chạy với 3.12.4), Windows. Cần có index Chroma của Day 6 (chạy Day 6 trước; mặc định ở
`%LOCALAPPDATA%/day06_chroma` vì Chroma 1.5.9 không mở lại được HNSW index dưới đường dẫn có dấu;
đổi bằng biến `DAY06_CHROMA_DIR`).

```bash
pip install -r requirements.txt
cp .env.example .env          # điền OPENAI_API_KEY; .env đã được .gitignore
```

## Chạy lại toàn bộ

Trên Windows đặt `PYTHONUTF8=1` (test gốc `test_baseline.py` đọc file bằng encoding mặc định cp1252).

```bash
export PYTHONUTF8=1
python -m unittest discover -s tests -v                    # 26 test, mock, không gọi API
python -m src.export_day6                                  # Day 6 Chroma -> data/corpus/chunks.json
python -m src.build_benchmark                              # -> data/benchmark/questions.json
python -m src.eval_retrieval --split dev --no-filter       # bảng retrieval dev
python -m src.run_agent --split dev                        # 9 cấu hình agent trên dev (~$0.05)
python -m src.label_dev baseline_dense rerank rerank_k5 rerank_multi rerank_retry
python starter/agent_metrics.py --cases results/agent/cases_dev.json \
  --traces results/agent/rerank/dev/traces.json --reviews results/agent/rerank/dev/reviews.json \
  --split dev --output results/agent/rerank/dev/agent_metrics.json
# --- chốt cấu hình (results/selection.json) rồi mới chạy test ---
python -m src.run_agent --split test --configs baseline_dense rerank
python -m src.eval_retrieval --split test
python -m src.label_test
python starter/agent_metrics.py --cases results/agent/cases_test.json \
  --traces results/agent/rerank/test/traces.json --reviews results/agent/rerank/test/reviews.json \
  --split test --output results/agent/rerank/test/agent_metrics.json
```

LLM chạy với `temperature=0` nhưng output OpenAI vẫn có thể khác nhẹ giữa các lần; trace gốc dùng cho báo cáo
nằm trong `results/`.
