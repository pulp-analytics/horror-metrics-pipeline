# Poster-metrics-pipeline: local entry points.
#
# `make sample` fills in *missing* files under OUT (default:
# data/sample_output). If the CSV already exists, there is no recipe --
# a clone (or CI checkout) is a no-op even when the scripts have newer
# mtimes than the checked-in sample. Delete a CSV to regenerate it.
# `make -j4 sample` is supported for an empty OUT.
#
# Intermediate caches (CLIP/SigLIP .npz) are prerequisites only when the
# CSV that needs them is itself missing. Otherwise a clone without the
# (uncommitted) .npz files would rebuild 05/11 and then overwrite the
# citable 06-09 / 12-13 sample. Same idea for 14→15 and 20+21→25.
#
# 22/23/24 (Nova QA) are not pipeline stages and are not in `sample`.

.PHONY: setup test test-fast color-sample sample assemble-sample help clean

IN     ?= data/sample_input/sample_100_posters.csv
OUT    ?= data/sample_output
PYTHON ?= python3

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo "Now: source .venv/bin/activate"

test:
	python3 -m pytest tests/ -v

test-fast:
	python3 -m pytest tests/ -v -m "not slow"

color-sample: $(OUT)/color_metrics.csv

help:
	@echo "make test-fast        pytest -m 'not slow' (what CI runs)"
	@echo "make sample           write missing $(OUT) files (01-21 + 25)"
	@echo "make -j4 sample       same, independent scripts in parallel"
	@echo "make assemble-sample  join $(OUT) into master_dataset.csv"
	@echo "  rebuild one metric by deleting its CSV, then make sample"
	@echo "  override paths with IN=... OUT=... PYTHON=..."

clean:
	rm -rf .venv __pycache__ scripts/__pycache__ scripts/utils/__pycache__ .pytest_cache

# --- sample graph -------------------------------------------------------
# Recipes are registered only when the output is missing (parse-time
# wildcard). Existing files must not depend on scripts/IN: GitHub
# Actions extracts scripts after data/, so script mtimes look newer
# than the checked-in CSVs and a timestamp graph would overwrite the
# sample.

SAMPLE_OUTPUTS = \
	$(OUT)/metrics_input.csv \
	$(OUT)/color_metrics.csv \
	$(OUT)/iqa_multi_score.csv \
	$(OUT)/nima_score.csv \
	$(OUT)/laion_aesthetic_score.csv \
	$(OUT)/census.csv \
	$(OUT)/fear_axis.csv \
	$(OUT)/typography.csv \
	$(OUT)/genre_classifier.csv \
	$(OUT)/medium.csv \
	$(OUT)/siglip_fear_axis.csv \
	$(OUT)/siglip_census.csv \
	$(OUT)/siglip_typography.csv \
	$(OUT)/siglip_genre_classifier.csv \
	$(OUT)/face_detect.csv \
	$(OUT)/face_expression.csv \
	$(OUT)/geometric_composition.csv \
	$(OUT)/depth_estimation.csv \
	$(OUT)/saliency_prediction.csv \
	$(OUT)/pose_dynamism.csv \
	$(OUT)/creature_weapon_owlv2.csv \
	$(OUT)/creature_weapon_dino.csv \
	$(OUT)/creature_weapon_agreement.csv

sample: $(SAMPLE_OUTPUTS)

assemble-sample:
	$(PYTHON) assemble_master_dataset.py --data-dir $(OUT) --out master_dataset.csv

ifeq ($(wildcard $(OUT)/metrics_input.csv),)
$(OUT)/metrics_input.csv: $(IN)
	mkdir -p $(OUT)
	cp $(IN) $@
endif

ifeq ($(wildcard $(OUT)/color_metrics.csv),)
$(OUT)/color_metrics.csv: $(IN) scripts/01_color_metrics.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/01_color_metrics.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/iqa_multi_score.csv),)
$(OUT)/iqa_multi_score.csv: $(IN) scripts/02_iqa_multi_score.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/02_iqa_multi_score.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/nima_score.csv),)
$(OUT)/nima_score.csv: $(IN) scripts/03_nima_score.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/03_nima_score.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/laion_aesthetic_score.csv),)
$(OUT)/laion_aesthetic_score.csv: $(IN) scripts/04_laion_aesthetic_score.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/04_laion_aesthetic_score.py --in $(IN) --out $@
endif

$(OUT)/clip_embeddings.npz: $(IN) scripts/05_clip_embed.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/05_clip_embed.py --in $(IN) --out $@

ifeq ($(wildcard $(OUT)/census.csv),)
$(OUT)/census.csv: $(OUT)/clip_embeddings.npz scripts/06_clip_census.py $(IN)
	$(PYTHON) scripts/06_clip_census.py --in $(IN) --embeddings $(OUT)/clip_embeddings.npz --out $@
endif

ifeq ($(wildcard $(OUT)/fear_axis.csv),)
$(OUT)/fear_axis.csv: $(OUT)/clip_embeddings.npz scripts/07_clip_fear_axis.py $(IN)
	$(PYTHON) scripts/07_clip_fear_axis.py --in $(IN) --embeddings $(OUT)/clip_embeddings.npz --out $@
endif

ifeq ($(wildcard $(OUT)/typography.csv),)
$(OUT)/typography.csv: $(OUT)/clip_embeddings.npz scripts/08_clip_typography_axis.py $(IN)
	$(PYTHON) scripts/08_clip_typography_axis.py --in $(IN) --embeddings $(OUT)/clip_embeddings.npz --out $@
endif

ifeq ($(wildcard $(OUT)/genre_classifier.csv),)
$(OUT)/genre_classifier.csv: $(OUT)/clip_embeddings.npz scripts/09_clip_genre_classifier.py $(IN)
	$(PYTHON) scripts/09_clip_genre_classifier.py --in $(IN) --embeddings $(OUT)/clip_embeddings.npz --out $@
endif

ifeq ($(wildcard $(OUT)/medium.csv),)
$(OUT)/medium.csv: $(IN) scripts/10_clip_medium.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/10_clip_medium.py --in $(IN) --out $@
endif

$(OUT)/siglip_embeddings.npz: $(IN) scripts/11_siglip_embed.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/11_siglip_embed.py --in $(IN) --out $@

ifeq ($(wildcard $(OUT)/siglip_fear_axis.csv),)
$(OUT)/siglip_fear_axis.csv: $(OUT)/siglip_embeddings.npz scripts/12_siglip_fear_axis.py $(IN)
	$(PYTHON) scripts/12_siglip_fear_axis.py --in $(IN) --embeddings $(OUT)/siglip_embeddings.npz --out $@
endif

# 13 writes three CSVs in one process. Make 3.81 has no grouped targets
# (`&:`), so census.csv is the recipe and the other two are aliases of it.
ifeq ($(wildcard $(OUT)/siglip_census.csv),)
$(OUT)/siglip_census.csv: $(OUT)/siglip_embeddings.npz scripts/13_siglip_reanalysis.py $(IN)
	$(PYTHON) scripts/13_siglip_reanalysis.py --in $(IN) --embeddings $(OUT)/siglip_embeddings.npz \
		--census-out $(OUT)/siglip_census.csv \
		--typography-out $(OUT)/siglip_typography.csv \
		--genre-out $(OUT)/siglip_genre_classifier.csv
endif

ifeq ($(wildcard $(OUT)/siglip_typography.csv),)
$(OUT)/siglip_typography.csv: $(OUT)/siglip_census.csv
endif

ifeq ($(wildcard $(OUT)/siglip_genre_classifier.csv),)
$(OUT)/siglip_genre_classifier.csv: $(OUT)/siglip_census.csv
endif

ifeq ($(wildcard $(OUT)/face_detect.csv),)
$(OUT)/face_detect.csv: $(IN) scripts/14_face_detect.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/14_face_detect.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/face_expression.csv),)
$(OUT)/face_expression.csv: $(OUT)/face_detect.csv scripts/15_face_expression.py $(IN)
	$(PYTHON) scripts/15_face_expression.py --in $(IN) --faces $(OUT)/face_detect.csv --out $@
endif

ifeq ($(wildcard $(OUT)/geometric_composition.csv),)
$(OUT)/geometric_composition.csv: $(IN) scripts/16_geometric_composition.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/16_geometric_composition.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/depth_estimation.csv),)
$(OUT)/depth_estimation.csv: $(IN) scripts/17_depth_estimation.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/17_depth_estimation.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/saliency_prediction.csv),)
$(OUT)/saliency_prediction.csv: $(IN) scripts/18_saliency_prediction.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/18_saliency_prediction.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/pose_dynamism.csv),)
$(OUT)/pose_dynamism.csv: $(IN) scripts/19_pose_dynamism.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/19_pose_dynamism.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/creature_weapon_owlv2.csv),)
$(OUT)/creature_weapon_owlv2.csv: $(IN) scripts/20_creature_weapon_owlv2.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/20_creature_weapon_owlv2.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/creature_weapon_dino.csv),)
$(OUT)/creature_weapon_dino.csv: $(IN) scripts/21_creature_weapon_dino.py
	mkdir -p $(OUT)
	$(PYTHON) scripts/21_creature_weapon_dino.py --in $(IN) --out $@
endif

ifeq ($(wildcard $(OUT)/creature_weapon_agreement.csv),)
$(OUT)/creature_weapon_agreement.csv: $(OUT)/creature_weapon_owlv2.csv $(OUT)/creature_weapon_dino.csv scripts/25_creature_weapon_agreement.py $(IN)
	$(PYTHON) scripts/25_creature_weapon_agreement.py --in $(IN) \
		--owlv2 $(OUT)/creature_weapon_owlv2.csv \
		--dino $(OUT)/creature_weapon_dino.csv \
		--out $@
endif
