MODEL_NAME ?= pp_doclayout_v3_preprocess_python
VERSION ?= 1
OUT ?= build/$(MODEL_NAME)

.PHONY: build clean

build:
	@test -n "$(OUT)"
	@test -n "$(MODEL_NAME)"
	@mkdir -p "$(OUT)/$(VERSION)"
	@sed "s/@MODEL_NAME@/$(MODEL_NAME)/g" config.pbtxt.in > "$(OUT)/config.pbtxt"
	@cp model.py "$(OUT)/$(VERSION)/model.py"

clean:
	rm -rf build
