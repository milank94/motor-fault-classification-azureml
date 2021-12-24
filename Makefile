create-register-environment:
	@echo 'Creating and registering training environment'
	@python scripts/create_environment.py

training-pipeline:
	@echo 'Run training pipeline'
	@python scripts/pipeline.py
