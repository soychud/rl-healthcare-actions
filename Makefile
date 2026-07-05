.PHONY: test lint clean

test:
	python3 -m pytest tests/ -v

lint:
	ruff check src/

clean:
	rm -rf __pycache__ .*.swp
