# Quy Ước Git (Git Conventions)

Tài liệu này định nghĩa toàn bộ quy chuẩn quản lý mã nguồn, cấu trúc commit, chiến lược nhánh (branching) và quy trình làm việc với Git cho dự án **Second Brain**.  
Tất cả các nhà phát triển (Developers) và **AI Coding Agents** bắt buộc phải tuân thủ nghiêm ngặt các quy tắc này.

---

## 1. Nguyên Tắc Cốt Lõi (Core Principles)

1. **Atomic Commits (Tính đơn mục tiêu):** Mỗi commit chỉ giải quyết một vấn đề, sửa một lỗi hoặc thêm một tính năng cụ thể. Tránh gộp các thay đổi không liên quan vào cùng một commit.
2. **Tuân thủ Conventional Commits:** Toàn bộ commit message phải theo chuẩn [Conventional Commits v1.0.0](https://www.conventionalcommits.org/).
3. **Tuyệt đối không commit dữ liệu nhạy cảm & dữ liệu cục bộ:**
   - Không commit file cấu hình chứa API Keys / Secrets (`backend/.env`, credentials).
   - Không commit dữ liệu tri thức cá nhân hoặc SQLite DB trong `storage/` hoặc `backend/storage/`.
   - Không commit thư mục build và cache (`node_modules/`, `.next/`, `__pycache__/`, `.venv/`).
4. **Kiểm tra trạng thái trước khi commit:** Luôn dùng `git status` và `git diff` để rà soát thay đổi trước khi stage. Không sử dụng `git add .` bừa bãi.

---

## 2. Định Dạng Commit Message (Commit Message Specification)

### 2.1. Cấu trúc tổng quát

```text
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

- **`<type>`**: Loại thay đổi (bắt buộc).
- **`(<scope>)`**: Phạm vi module/thành phần bị tác động (tùy chọn nhưng khuyến khích).
- **`<subject>`**: Mô tả ngắn gọn về thay đổi (bắt buộc).
- **`<body>`**: Giải thích chi tiết động cơ, lý do thay đổi (nếu cần).
- **`<footer>`**: Thông tin Breaking Changes hoặc liên kết Issue/PR (nếu có).

---

### 2.2. Danh sách Commit Types

| Type | Ý nghĩa | Khi nào sử dụng |
| :--- | :--- | :--- |
| `feat` | Tính năng mới | Thêm mới chức năng, endpoint, UI component, dịch vụ |
| `fix` | Sửa lỗi | Vá lỗi logic, sửa exception, sửa hiển thị sai |
| `docs` | Tài liệu | Thay đổi tài liệu hướng dẫn, README, docs trong `docs/` |
| `refactor` | Tái cấu trúc | Thay đổi mã nguồn mà không sửa bug hay thêm tính năng mới |
| `perf` | Hiệu năng | Tối ưu tốc độ xử lý, giảm thời gian truy vấn, tối ưu memory |
| `test` | Kiểm thử | Thêm mới hoặc cập nhật bộ unit test, integration test |
| `build` | Hệ thống build | Thay đổi cấu hình gói phụ thuộc (`pyproject.toml`, `package.json`, Dockerfile) |
| `ci` | CI / CD | Cập nhật cấu hình GitHub Actions, kịch bản CI tự động |
| `style` | Phong cách mã | Sửa format code, khoảng trắng, thụt đầu dòng (không ảnh hưởng logic) |
| `chore` | Việc phụ trợ | Cập nhật `.gitignore`, dọn dẹp file tạm, nâng cấp script phụ |
| `revert` | Hoàn tác | Revert lại một commit trước đó |

---

### 2.3. Danh sách Scopes trong dự án

Dự án **Second Brain** chia phạm vi theo các thành phần chính:

| Scope | Khu vực mã nguồn |
| :--- | :--- |
| `backend` | FastAPI server, routes, core configuration, database |
| `frontend` | Next.js app, UI components, Tailwind styles, state management |
| `retrieval` | Bộ quét văn bản Markdown (Markdown Scanner), Query Expander, Citation |
| `ingestion` | Bộ phân tích tài liệu (PDF, Word, Markdown, URL scraper) |
| `llm` | Bộ kết nối LLM (LiteLLM, Prompt templates, Multi-provider) |
| `api` | FastAPI routers, Pydantic schemas, validation |
| `vault` | Định dạng & lưu trữ Markdown Vault, Obsidian compatibility |
| `skills` | Kỹ năng mở rộng cho AI Agent (`.agents/skills/`) |
| `docker` | Cấu hình Docker, `docker-compose.yml`, Dockerfiles |
| `system` | Tích hợp tổng thể nhiều thành phần |

---

### 2.4. Quy tắc viết `<subject>`

- Viết bằng **chữ thường (lowercase)**, không viết hoa chữ cái đầu.
- Sử dụng **thể mệnh lệnh (imperative mood)** (ví dụ: `implement` thay vì `implemented` hoặc `implements`, `add` thay vì `added`).
- **Không có dấu chấm (`.`)** ở cuối dòng subject.
- Độ dài subject tối đa **72 ký tự**.

#### Ví dụ chuẩn:
```text
feat(retrieval): implement direct markdown span scanner with citations
fix(backend): register models in app startup and resolve deprecation warnings
refactor: completely remove chromadb in favor of pure markdown vault search
docs(ai): update git conventions for project agents
chore: ignore storage directories in git
test(retrieval): add unit tests for query expansion algorithm
```

#### Ví dụ không hợp lệ:
```text
❌ fixed bug                      (thiếu type, không cụ thể)
❌ Feat: Add new feature.         (viết hoa, có dấu chấm ở cuối)
❌ update code                    (không rõ mục đích, không theo chuẩn)
❌ git commit -m "done"           (vô nghĩa)
```

---

## 3. Chiến Lược Nhánh (Branching Strategy)

Dự án áp dụng mô hình nhánh rút gọn, tập trung vào tính linh hoạt và an toàn:

### 3.1. Các loại nhánh

- **`master` / `main`**: Nhánh chính, phản ánh trạng thái ổn định (production-ready). Mã nguồn trên nhánh này phải luôn chạy được và vượt qua toàn bộ tests.
- **`feat/<tên-ngắn>`**: Nhánh phát triển tính năng mới (ví dụ: `feat/pdf-ocr`, `feat/export-notes`).
- **`fix/<tên-ngắn>`**: Nhánh sửa lỗi (ví dụ: `fix/cors-origin`, `fix/citation-line-range`).
- **`refactor/<tên-ngắn>`**: Nhánh cấu trúc lại mã (ví dụ: `refactor/retrieval-service`).
- **`docs/<tên-ngắn>`**: Nhánh cập nhật tài liệu dự án.

### 3.2. Quy tắc đặt tên nhánh
- Sử dụng chữ thường, phân cách các từ bằng dấu gạch ngang (`-`).
- Ngắn gọn nhưng phản ánh đúng mục tiêu (ví dụ: `feat/session-save-skill`).

---

## 4. Quy Trình Làm Việc Chuẩn (Standard Workflow)

1. **Cập nhật nhánh mới nhất:**
   ```bash
   git checkout master
   git pull origin master
   ```

2. **Tạo nhánh làm việc (nếu thực hiện thay đổi lớn):**
   ```bash
   git checkout -b feat/<tên-tính-năng>
   ```

3. **Thực hiện thay đổi và kiểm thử:**
   - Chạy kiểm thử Backend: `pytest` hoặc `uv run pytest`
   - Kiểm tra linter / TypeScript Frontend: `npm run lint` / `npm run build`

4. **Kiểm tra trạng thái staging:**
   ```bash
   git status
   git diff
   ```

5. **Stage các tệp tin có chủ đích (Tránh `git add .` bừa bãi):**
   ```bash
   git add backend/app/services/retrieval.py
   git add docs/.ai/GIT_CONVENTIONS.md
   ```

6. **Tạo commit theo chuẩn:**
   ```bash
   git commit -m "feat(retrieval): add fuzzy keyword matching in markdown scanner"
   ```

---

## 5. Hướng Dẫn Đặc Biệt Dành Cho AI Coding Agents

Khi AI Agent thực hiện các thao tác Git:
1. **Kiểm tra `git status` trước khi commit:** Xác định danh sách file thay đổi thực tế; loại trừ các file log, scratch, artifact tạm thời.
2. **Không stage file nhạy cảm:** Không bao giờ stage file `.env`, file token, DB SQLite hay file trong `storage/`.
3. **Commit message chính xác:** Tự động tạo message theo đúng chuẩn Conventional Commits với type và scope phù hợp nhất với task vừa hoàn thành.
4. **Không force-push:** Tuyệt đối không dùng `git push --force` hoặc `git reset --hard` trừ khi có chỉ đạo rõ ràng từ người dùng.
5. **Đồng bộ hóa tài liệu agent:** Khi cập nhật các file quy tắc trong `docs/.ai/`, chạy script `bash scripts/generate_agents_markdown.sh` để đồng bộ nội dung vào `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`.
