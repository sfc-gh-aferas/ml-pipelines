# Snowflake MLOps Framework

A production-ready framework for deploying and managing machine learning workflows on Snowflake with automated CI/CD.

## Overview

This framework enables teams to:
- Deploy ML workflows as scheduled Snowflake DAGs (Tasks)
- Manage features with a versioned feature store
- Automate deployments via GitHub Actions CI/CD
- Share utilities through the `ml_utils` package

## Getting Started

This repo is intended to be used as an example. **Copy this directory into your own repository** before running any GitHub Actions or making modifications. The CI/CD workflow is self-contained and will work once you configure the required secrets in your repo.

### Prerequisites
- Python 3.10
- A Snowflake account with ACCOUNTADMIN access (for initial setup)
- [uv](https://docs.astral.sh/uv/) package manager
- A Snowflake connection configured in `~/.snowflake/connections.toml`

### Installation

```bash
# Install dependencies and build the shared ml_utils package
uv sync --locked
uv build
```

---

## Environment Setup

The framework uses three isolated environments. Each environment has its own database, role, warehouses, and service users.

| Environment | Database | Role | Usage |
|-------------|----------|------|-------|
| `DEV` | `ML_PIPELINE_DEV_DB` | `ML_PIPELINE_DEV_ROLE` | Local development and experimentation |
| `STAGING` | `ML_PIPELINE_STAGING_DB` | `ML_PIPELINE_STAGING_ROLE` | CI/CD deploys from feature branches |
| `PROD` | `ML_PIPELINE_PROD_DB` | `ML_PIPELINE_PROD_ROLE` | CI/CD deploys from main branch |

### Creating DEV, STAGING, and PROD Environments

The `setup/setup.sql` script creates all required Snowflake resources for a given environment. Run it three times, once per environment, replacing the `<% env %>` placeholder with `DEV`, `STAGING`, or `PROD`.

For each environment the script will:
- Create the environment role (`ML_PIPELINE_<env>_ROLE`) and grant it required account-level privileges
- Grant the role to the current user
- Create the database (`ML_PIPELINE_<env>_DB`) with `BASE_DATA` and `ML_PROJECTS` schemas
- Create `DATA_STAGE`, `BUILD_STAGE`, and `JOB_STAGE` stages
- Create a default XS warehouse
- Create a service user (`GIT_ACTIONS_<env>`) for CI/CD automation
- Load the included demo dataset (`MORTGAGE_LENDING_DEMO_DATA`)

**Steps:**

Use the [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index) (`snow sql`) to execute the setup script. The `-D` flag substitutes the `<% env %>` template variable:

```bash
# Create the DEV environment
snow sql -f setup/setup.sql -D "env=DEV" -c <your_connection>

# Create the STAGING environment
snow sql -f setup/setup.sql -D "env=STAGING" -c <your_connection>

# Create the PROD environment
snow sql -f setup/setup.sql -D "env=PROD" -c <your_connection>
```

Replace `<your_connection>` with the name of a connection in `~/.snowflake/connections.toml` that has ACCOUNTADMIN access.

---

## Creating PATs and Configuring GitHub Secrets

The CI/CD pipeline authenticates to Snowflake using the `GIT_ACTIONS_STAGING` and `GIT_ACTIONS_PROD` service users. You need to generate a Programmatic Access Token (PAT) for each.

### 1. Generate PATs in Snowflake

For each service user (`GIT_ACTIONS_STAGING` and `GIT_ACTIONS_PROD`), generate a password or PAT:

```sql
-- As ACCOUNTADMIN or USERADMIN
ALTER USER GIT_ACTIONS_STAGING SET PASSWORD = '<strong-password>';
ALTER USER GIT_ACTIONS_PROD SET PASSWORD = '<strong-password>';
```

Or, if using key-pair authentication, generate and assign RSA keys per Snowflake documentation.

### 2. Add GitHub Secrets

In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add the following secrets:

| Secret Name | Value |
|---|---|
| `SNOWFLAKE_ACCOUNT` | Your Snowflake account identifier (e.g. `myorg-myaccount`) |
| `SNOWFLAKE_STAGING_USER` | `GIT_ACTIONS_STAGING` |
| `SNOWFLAKE_STAGING_PASSWORD` | Password/PAT for the STAGING service user |
| `SNOWFLAKE_PROD_USER` | `GIT_ACTIONS_PROD` |
| `SNOWFLAKE_PROD_PASSWORD` | Password/PAT for the PROD service user |

The deploy workflow (`.github/workflows/deploy.yml`) automatically selects the correct user and password based on the branch:
- **Push to `main`** uses `SNOWFLAKE_PROD_USER` / `SNOWFLAKE_PROD_PASSWORD`
- **Push to any other branch** uses `SNOWFLAKE_STAGING_USER` / `SNOWFLAKE_STAGING_PASSWORD`

---

## Local Development

Set these environment variables for local development:

```bash
export SNOWFLAKE_CONNECTION="your_connection_name"  # From connections.toml (connection must have role defined)
export SNOWFLAKE_ENVIRONMENT="DEV"
```

### Deploy Feature Store Locally

```bash
python feature_store/setup_feature_store.py
```

This registers entities and feature views defined in `feature_store/config.yml`, creates warehouses, and applies environment-appropriate privileges. Feature views are automatically versioned based on their definition; only breaking changes (query, entities, schema) increment the version.

### Deploy a Project Locally

```bash
# Deploy only (creates resources, uploads code, schedules DAGs)
python scripts/deploy_project.py example_project

# Deploy and immediately execute all DAGs (useful for testing)
python scripts/deploy_project.py example_project --run-dag
```

### Create a New Project

```bash
./scripts/create_project.sh my_project
```

This copies the `template/` directory into `projects/my_project` and generates:
- `config.yml` - DAG and compute resource configuration
- `utils.py` - Project-level utilities (stage path helpers)
- `pip-requirements.txt` - Python dependencies for container notebooks and ML Jobs

### Clean Up Resources

```bash
# Preview what would be deleted
python scripts/cleanup.py example_project --dry-run

# Delete all resources for a project
python scripts/cleanup.py example_project

# Delete specific feature views
python scripts/cleanup.py --features example_features --dry-run
python scripts/cleanup.py --features example_features

# Delete everything (all projects, features, stages)
python scripts/cleanup.py --all --dry-run
python scripts/cleanup.py --all
```

---

## Feature Store

### Configuration

Define entities and feature views in `feature_store/config.yml`:

```yaml
entities:
  - name: loan_entity
    join_keys:
      - LOAN_ID

warehouses:
  - name: FEATURE_STORE
    warehouse_size: SMALL
    auto_suspend: 300
    default: true

feature_views:
  - name: example_features
    function: create_example_features   # Function in feature_views.py returning a Snowpark DataFrame
    entities: loan_entity               # Entity name or list of entity names
    timestamp_col: TIMESTAMP
    desc: Example features
    # warehouse: HEAVY_COMPUTE          # Optional non-default warehouse
    # refresh_freq: 1 day               # Optional auto-refresh
```

### Implementing Feature Views

Add feature generation functions in `feature_store/feature_views.py`. Each function receives a Snowpark `Session` and must return a Snowpark `DataFrame`:

```python
from snowflake.snowpark import Session, DataFrame

def create_example_features(session: Session) -> DataFrame:
    df = session.table("BASE_DATA.MORTGAGE_LENDING_DEMO_DATA")
    # ... feature engineering ...
    return df.select("LOAN_ID", "TIMESTAMP", "FEATURE1", "FEATURE2")
```

### Versioning

Feature views are automatically versioned. Breaking changes (query logic, entities, timestamp column, clustering) increment the version number. Non-breaking changes (refresh frequency, warehouse, description) update the existing version in-place.

---

## Creating ML Projects

### Project Configuration

Edit `projects/<project>/config.yml` to define compute resources and DAGs:

```yaml
project_name: my_project
active: True  # Set to True to enable deployment

deploy:
  warehouse:
    warehouse_size: SMALL
    auto_suspend: 300
  compute_pool:
    instance_family: CPU_X64_XS
  DAGS:
    - name: TRAINING_PIPELINE
      tasks:
        - name: prepare_data
          file: prepare_data.py          # Runs on warehouse (direct Python execution)
        - name: train_model
          file: training.py
          mljob: True                    # Runs as ML Job on SPCS compute pool
          dep: prepare_data
        - name: evaluate
          file: evaluate.ipynb           # Runs as Snowflake Notebook
          dep: train_model
      schedule: CRON 0 3 * * * UTC
      conda_packages:
        - xgboost
```

### Task Types

| Type | Config | Execution |
|------|--------|-----------|
| Warehouse Python | `.py` file, `mljob: false` (default) | Runs `main(session, **kwargs)` directly on warehouse |
| ML Job (SPCS) | `.py` file, `mljob: true` | Runs as containerized ML Job on compute pool |
| Notebook | `.ipynb` file | Deploys and executes as Snowflake Notebook Project |

### Schedule Formats
- **CRON:** `CRON 0 9 * * * UTC` (daily at 9 AM)
- **Interval:** `3 HOURS`, `30 MINUTES`, `5 SECONDS`
- **Empty/null:** Manual trigger only

### Passing Data Between Tasks

Tasks can pass data to downstream tasks via return values (must be a `dict`). Downstream tasks declare dependencies via the `dep` field and receive predecessor return values as keyword arguments (warehouse tasks) or CLI arguments (ML Jobs).

---

## CI/CD Pipeline

### How It Works

The GitHub Actions workflow (`.github/workflows/deploy.yml`) triggers on any push that changes files in `feature_store/` or `projects/`.

**Pipeline steps:**
1. **Checkout** code with `fetch-depth: 2` for change detection
2. **Install** uv, Python, dependencies, and build the `ml_utils` wheel
3. **Detect changes** by diffing `HEAD^ HEAD` to identify modified feature store files and project directories
4. **Deploy Feature Store** if `feature_store/` files changed
5. **Deploy Projects** for each project directory with changes (runs with `--run-dag` to validate)
6. **Deployment Summary** posted to the GitHub Actions step summary

### Environment Routing

| Trigger | Environment | Snowflake User |
|---------|-------------|----------------|
| Push to `main` | `PROD` | `GIT_ACTIONS_PROD` |
| Push to any other branch | `STAGING` | `GIT_ACTIONS_STAGING` |

### Change Detection

The pipeline only deploys what changed:
- Changes in `feature_store/` trigger feature store redeployment
- Changes in `projects/<name>/` trigger redeployment of that specific project
- Multiple projects can be deployed in a single run

### Deployment Behavior by Environment

| Behavior | DEV (local) | STAGING (CI/CD) | PROD (CI/CD) |
|----------|-------------|-----------------|--------------|
| Feature views | Active | Suspended after deploy | Active with schedule |
| DAGs | Suspended after deploy | Suspended after deploy (run once for validation) | Active with schedule |
| Privileges | Full access | Read-only / monitor | Read-only / monitor |

---

## Repository Structure

```
ml-pipelines/
├── .github/workflows/
│   └── deploy.yml              # CI/CD pipeline
├── feature_store/
│   ├── config.yml              # Entity, warehouse, and feature view definitions
│   ├── feature_views.py        # Feature generation functions
│   └── setup_feature_store.py  # Feature store deployment script
├── projects/
│   └── example_project/
│       ├── config.yml           # DAG and compute config
│       ├── training.py          # ML Job script
│       ├── training.ipynb       # Notebook task
│       ├── inference.py         # Warehouse task
│       ├── prepare_data.py      # Data preparation
│       ├── promote_model.py     # Model promotion
│       ├── utils.py             # Project utilities
│       └── pip-requirements.txt # SPCS/notebook dependencies
├── template/                    # New project template
├── ml_utils/                    # Shared utilities package
│   ├── snowflake_env.py         # Environment config and session management
│   └── utils.py                 # Shared helpers (e.g., version_data)
├── scripts/
│   ├── create_project.sh        # Create new project from template
│   ├── deploy_project.py        # Deploy project to Snowflake
│   └── cleanup.py               # Remove deployed resources
├── setup/
│   ├── setup.sql                # Environment bootstrap SQL
│   └── MORTGAGE_LENDING_DEMO_DATA.csv.gz
├── pyproject.toml               # Package definition
└── dist/                        # Built ml_utils wheel
```

## Troubleshooting

**Task fails with import error**
- Ensure `pip-requirements.txt` includes all dependencies
- Rebuild `ml_utils`: `uv build`
- Redeploy the project

**DAG doesn't run on schedule**
- Check the root task is in STARTED state: `SHOW TASKS IN SCHEMA;`
- Verify schedule format in `config.yml`
- Confirm the environment is PROD (non-PROD DAGs are suspended after deploy)

**Feature store entity not found**
- Entities must be registered before feature views
- Verify entity names match exactly in `config.yml`

**CI/CD deploy fails with authentication error**
- Verify all five GitHub secrets are set correctly
- Ensure service users exist and have the correct roles granted
- Check network policy allows GitHub Actions runner IPs

## Resources

- [Snowflake Tasks & DAGs](https://docs.snowflake.com/en/user-guide/tasks-intro)
- [Snowflake ML Jobs](https://docs.snowflake.com/en/developer-guide/snowpark-ml/ml-jobs)
- [Snowflake Feature Store](https://docs.snowflake.com/en/developer-guide/snowpark-ml/feature-store/overview)
- [Snowflake Notebook Projects](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks)
