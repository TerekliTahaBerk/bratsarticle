FROM python:3.11.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace/bratsarticle

COPY environment/q1q2_v2-requirements-lock.txt /tmp/requirements-lock.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-lock.txt

COPY . /workspace/bratsarticle
RUN python -m pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "pytest", "-q"]
