IMAGE=affinefdn/validator:latest

.PHONY: test build push

test:
	pytest -q

build:
	docker build -t $(IMAGE) .

push: build
	docker push $(IMAGE) 