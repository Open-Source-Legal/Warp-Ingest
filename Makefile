# Variables
DOCKER_IMAGE_NAME = warp-ingest-test
DOCKER_TAG = latest
CONTAINER_NAME = warp-ingest-test-container
HOST_PORT = 5010
CONTAINER_PORT = 5001

# Development Commands
.PHONY: format lint check run_fmt generate_requirements download_nltk_data
format:
	uv run black .
	uv run isort .

lint:
	uv run black --check .
	uv run isort --check .

run_fmt: format

check: lint

generate_requirements:
	uv export --format requirements.txt --extra ocr --group dev --output-file requirements.txt

download_nltk_data:
	uv run python -m nltk.downloader punkt punkt_tab stopwords

# Benchmark Commands — official LlamaIndex ParseBench (see benchmarks/parsebench/)
.PHONY: parsebench-setup parsebench parsebench-full
parsebench-setup:
	@echo "Installing the official ParseBench framework + baseline parsers (one-time)..."
	bash benchmarks/parsebench/setup_parsebench.sh

parsebench:
	@echo "Running ParseBench --test (3 files/category) for warp-ingest + baselines..."
	python -m benchmarks.parsebench.run --test

parsebench-full:
	@echo "Running the full, leaderboard-comparable ParseBench benchmark..."
	python -m benchmarks.parsebench.run --full

# Test Commands  (no Java / Tika needed -- the parser is pure Python)
.PHONY: test test-pdf-ingestor test-s1 test-s1-full
test:
	@echo "Running all tests..."
	uv run pytest -vvs .

test-pdf-ingestor:
	@echo "Running PDF ingestor tests..."
	PYTHONPATH=. uv run python tests/run_ingestor_page_test.py

# S-1 cross-engine regression vs the original Java/Tika nlm-ingestor.
# `test-s1` runs the fast subset; `test-s1-full` adds the large multi-hundred-page bodies.
test-s1:
	PYTHONPATH=. uv run pytest tests/test_s1_regression.py -q

test-s1-full:
	PYTHONPATH=. uv run pytest tests/test_s1_regression.py --runslow -q

# Docker Commands
.PHONY: build run run-test-all
build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) -f Dockerfile.test .

run:
	@echo "Running Docker container..."
	docker run -d --name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		-v $(PWD)/files:/app/files \
		--add-host=host.docker.internal:host-gateway \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	@echo "Waiting for container to start..."
	@sleep 8

run-test-all: clean build run test clean

# Cleanup Commands
.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) || true
