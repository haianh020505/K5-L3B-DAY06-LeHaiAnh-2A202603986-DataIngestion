# Day 06 · Data Ingestion Engineering

> **Bài làm — Lê Hải Anh (2A202603986).** Báo cáo: [report/individual/2A202603986_LeHaiAnh.md](report/individual/2A202603986_LeHaiAnh.md)
>
> | | |
> |---|---|
> | Corpus | OpenStax *Introduction to Python Programming*, ch. 4–6, 68 trang — [data/manifest.json](data/manifest.json), thống kê từng trang [data/manifest_pages.csv](data/manifest_pages.csv) |
> | Benchmark | 20 câu (15 dev / 5 test, 5 câu có filter) — [benchmark/questions.json](benchmark/questions.json) |
> | Code | [starter/src/](starter/src/) — Chroma 1.5.9 (cosine, lưu đĩa) + `openai/text-embedding-3-small` qua OpenRouter |
> | Kết quả | [starter/results/](starter/results/) (trace, run log, nhãn audit), [artifacts/experiment_log.csv](artifacts/experiment_log.csv) |
>
> **Chạy lại** (Windows PowerShell, Python 3.13, không cần Docker):
> ```powershell
> cd starter
> py -3.13 -m venv .venv; .venv\Scripts\Activate.ps1
> pip install -r requirements.txt
> Copy-Item .env.example .env   # điền OPENROUTER_API_KEY, EMBEDDING_MODEL=openai/text-embedding-3-small
> # tải PDF từ https://openstax.org/details/books/introduction-python-programming
> # lưu thành starter/data/Introduction_to_Python_Programming-WEB.pdf (PDF không được commit, xem giấy phép)
> python -m src.run
> ```
> Lệnh trên đọc PDF, kiểm chứng benchmark, nạp lại index A/B vào `starter/chroma_db/`, chạy A/B/C rồi ghi đè `starter/results/` và `artifacts/experiment_log.csv`. File `audit_*.csv` đã có (chứa nhãn thủ công) không bị ghi đè; mẫu mới được ghi ra `audit_*.new.csv`.

---

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
