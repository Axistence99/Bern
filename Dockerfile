# Bern in a container, for sharing a link instead of a repo.
#
#   docker build -t bern .
#   docker run -p 8000:8000 bern
#
# Then open http://localhost:8000, or give a friend your machine's address on the same
# network. Set BERN_LLM_API_KEY at run time to enable the optional language model:
#
#   docker run -p 8000:8000 -e BERN_LLM_API_KEY=sk-... bern

FROM python:3.13-slim

# OpenCV needs these two even in headless mode: the wheel links against libGL and libglib.
# Without them `import cv2` fails at runtime with a confusing shared-library error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source change does not invalidate the wheel layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# models/bern_model.joblib is gitignored, so a fresh checkout does not contain it and the
# first request would pay ~10 s to train it. Training at build time keeps container start
# instant and moves the cost to where it is paid once.
RUN python3 -m bern.trainer --fast

# Run as a non-root user. The app writes nothing outside its own tree, so this costs nothing.
RUN useradd --create-home --uid 10001 bern && chown -R bern:bern /app
USER bern

ENV PORT=8000
EXPOSE 8000

# The app reads PORT from the environment and binds 0.0.0.0, which is what a container needs.
CMD ["python3", "server.py"]
