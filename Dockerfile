# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
WORKDIR /app

# apt 캐시 재사용
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 의존성 레이어(최대 캐시효과) - requirements만 먼저 복사
COPY requirements.txt .

# pip 캐시 재사용 (다음 빌드부터 다운로드 거의 없음)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 소스는 마지막에
COPY . .
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
