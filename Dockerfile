FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.2 /uv /uvx /bin/

COPY uv.lock pyproject.toml /

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV TZ=Europe/Moscow
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV FORCE_COLOR=1

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . .

CMD ["uv", "run", "--no-dev", "python", "main.py"]
