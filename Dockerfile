FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8050
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements-lab.txt ./
RUN pip install --no-cache-dir -r requirements-lab.txt
COPY . .
RUN useradd --create-home lab && chown -R lab:lab /app
USER lab
EXPOSE 8050
CMD ["sh", "-c", "exec gunicorn cacib_lab.app:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 2 --timeout 60"]
