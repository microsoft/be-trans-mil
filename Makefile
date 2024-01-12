.ONESHELL:  # Run all commands inside the same shell
# This ensures 'pip install' executes in the activated conda environment

# This is necessary because 'conda activate' doesn't work out-of-the-box in the make shell
# Adapted from https://stackoverflow.com/a/55696820
conda_activate := . $(shell conda info --base)/etc/profile.d/conda.sh ; conda activate
conda_deactivate := . $(shell conda info --base)/etc/profile.d/conda.sh ; conda deactivate

conda_env_name = HimlHisto

# Create a Conda environment for this folder only.
# We are re-using the HimlHisto environment without changes, via a symbolic link.
env:
	$(conda_deactivate)
	conda env remove --name $(conda_env_name)
	conda env create --file environment.yml --name $(conda_env_name)
	$(conda_activate) $(conda_env_name)

# Update the Conda environment
env_update:
	conda env update -n $(conda_env_name) --file environment.yml --prune

# Lock the current Conda environment secondary dependencies versions
lock_env:
	hi-ml/create_and_lock_environment.sh

# clean build artifacts
clean:
	for folder in .mypy_cache __pycache__ logs outputs; do \
		rm -rf `find . -type d -name $$folder`; \
	done
	rm -rf ./.pytest_cache
	rm -rf htmlcov
	rm coverage.xml .coverage

# pip upgrade and install test requirements
pip_test:
	python -m pip install --upgrade pip
	pip install -r requirements_test.txt

# Run black formatting on the codebase
black:
	black cyted testcyted *.py

# run flake8, assuming test requirements already installed
flake8:
	flake8 --count --statistics .

# run mypy, assuming test requirements already installed
mypy:
	mypy --install-types --show-error-codes --non-interactive --package cyted
	mypy --install-types --show-error-codes --non-interactive --package testcyted

# run basic checks
check: flake8 mypy

# run pytest on package, assuming test requirements already installed
pytest:
	pytest

# Run pytest with coverage on package
# Output the slowest tests via the --durations flag.
pytest_coverage:
	pytest --durations=20 --cov=cyted --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc

# run pytest using GPUs via an AzureML job
pytest_gpu:
	python run_pytest.py \
		--mark=gpu \
		--cluster=smoke-tests-cluster \
		--conda_env=environment.yml \
		--folder=testcyted \
		--coverage_module=cyted \
		--strictly_aml_v1=True

# Mount the datasets container at /cyted
# When running on a Github agent, this will read the account key and account name from
# environment variables.
mount_blobfuse:
	setup/blobfuse_cfg_from_env.sh
	setup/mount_blob.sh
	rm blobfuse.cfg

# Run regression tests and compare performance

BASE_CPATH_RUNNER_COMMAND := python runner.py --mount_in_azureml

DEFAULT_SMOKE_TEST_ARGS := --pl_limit_train_batches=2 --pl_limit_val_batches=4 --pl_limit_test_batches=4 --max_epochs=1

define DEFAULT_SMOKE_DEEPMIL_ARGS
--smoke_run=True --max_num_workers=6 \
--max_bag_size=3 --max_bag_size_inf=3 --batch_size=2 --batch_size_inf=2 --encoding_chunk_size=3 \
--crossval_count=0 --num_top_slides=2 --num_top_tiles=2 --pl_log_every_n_steps=5
endef

CYTED_AML_COMPUTE_ARGS := --cluster=smoke-tests-cluster --wait_for_completion --strictly_aml_v1

MIL_REGRESSION_TEST_METRICS := --regression_metrics='train/loss_epoch,val/loss_epoch,extra_val/loss_epoch,test/loss_epoch' --regression_test_csv_tolerance=1e-5

TFF3_HECytedMIL := --model=cyted.TFF3_HECytedMIL --regression_test_folder=testcyted/testdata/smoke_test_tff3_he_cyted ${MIL_REGRESSION_TEST_METRICS}

TFF3CytedMIL := --model=cyted.TFF3CytedMIL --regression_test_folder=testcyted/testdata/smoke_test_tff3_cyted ${MIL_REGRESSION_TEST_METRICS}

TFF3_HE_Cyted_MIL_ARGS := ${BASE_CPATH_RUNNER_COMMAND} ${TFF3_HECytedMIL} ${DEFAULT_SMOKE_TEST_ARGS} ${DEFAULT_SMOKE_DEEPMIL_ARGS}

TFF3_Cyted_MIL_ARGS := ${BASE_CPATH_RUNNER_COMMAND} ${TFF3CytedMIL} ${DEFAULT_SMOKE_TEST_ARGS} ${DEFAULT_SMOKE_DEEPMIL_ARGS}

TFF3_SIMCLR_ARGS := ${BASE_CPATH_RUNNER_COMMAND} --model=cyted.TFF3CytedSimCLR --ssl_training_batch_size=5 ${DEFAULT_SMOKE_TEST_ARGS}

HE_SIMCLR_ARGS := ${BASE_CPATH_RUNNER_COMMAND} --model=cyted.HECytedSimCLR --ssl_training_batch_size=5 ${DEFAULT_SMOKE_TEST_ARGS}

smoke_test_tff3_he_cyted_local:
	{
		${TFF3_HE_Cyted_MIL_ARGS}
	}

smoke_test_tff3_he_cyted_aml:
	{
		${TFF3_HE_Cyted_MIL_ARGS} \
		${CYTED_AML_COMPUTE_ARGS} \
		--tag=smoke_test_tff3_he_cyted \
		;
	}

smoke_test_tff3_cyted_local:
	{
		${TFF3_Cyted_MIL_ARGS}
	}

smoke_test_tff3_cyted_aml:
	{
		${TFF3_Cyted_MIL_ARGS} \
		${CYTED_AML_COMPUTE_ARGS} \
		--tag=smoke_test_tff3_cyted \
		;
	}

smoke_test_cyted_tff3_simclr_local:
	{
		${TFF3_SIMCLR_ARGS}
	}

smoke_test_cyted_tff3_simclr_aml:
	{
		${TFF3_SIMCLR_ARGS} \
		${CYTED_AML_COMPUTE_ARGS} \
		--tag=smoke_test_cyted_tff3_simclr;
	}

smoke_test_cyted_he_simclr_local:
	{
		${HE_SIMCLR_ARGS}
	}

smoke_test_cyted_he_simclr_aml:
	{
		${HE_SIMCLR_ARGS} \
		${CYTED_AML_COMPUTE_ARGS} \
		--tag=smoke_test_cyted_he_simclr;
	}
