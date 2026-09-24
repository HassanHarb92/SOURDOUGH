.PHONY: install test inventory app tree

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

inventory:
	PYTHONPATH=$$PWD/src python scripts/inspect_all_h5.py

app:
	PYTHONPATH=$$PWD/src streamlit run app/streamlit_app.py

tree:
	find . -maxdepth 3 -type f | sort

schema:
	PYTHONPATH=$$PWD/src python scripts/compare_h5_schema.py

semantic:
	PYTHONPATH=$$PWD/src python scripts/probe_xrf_semantics.py

validate-scans:
	PYTHONPATH=$$PWD/src python scripts/validate_xrf_scans.py

validate-explorer:
	PYTHONPATH=$$PWD/src python scripts/validate_xrf_explorer.py

validate-qc:
	PYTHONPATH=$$PWD/src python scripts/validate_qc_normalization.py

validate-intensity:
	PYTHONPATH=$$PWD/src python scripts/validate_intensity_contrast.py

validate-gradients:
	PYTHONPATH=$$PWD/src python scripts/validate_gradients.py

validate-hessian:
	PYTHONPATH=$$PWD/src python scripts/validate_hessian.py

validate-landscape:
	PYTHONPATH=$$PWD/src python scripts/validate_cell_landscape.py

validate-cells:
	PYTHONPATH=$$PWD/src python scripts/validate_cell_analyzer.py

analyze-directory:
	PYTHONPATH=$$PWD/src python scripts/analyze_directory.py img.dat

validate-quantification:
	PYTHONPATH=$$PWD/src python scripts/validate_beam_quantification.py

validate-concentration:
	PYTHONPATH=$$PWD/src python scripts/validate_maps_concentration.py

analyze-study:
	PYTHONPATH=$$PWD/src python scripts/analyze_study.py img.dat

validate-study:
	PYTHONPATH=$$PWD/src python scripts/validate_study_analysis.py

validate-artifacts:
	PYTHONPATH=$$PWD/src python scripts/validate_artifact_screening.py
