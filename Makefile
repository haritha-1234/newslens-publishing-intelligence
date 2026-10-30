.PHONY: all run app test clean

all: run

run:
	python main.py

app:
	streamlit run app.py

test:
	python -m pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "Cleaned up!"