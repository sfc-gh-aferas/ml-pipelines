# Snowflake MLOps Framework

A production-ready framework for deploying and managing machine learning workflows on Snowflake with automated CI/CD.

## Overview

This framework enables teams to:
- 🚀 **Deploy ML workflows** as scheduled Snowflake DAGs
- 🗄️ **Manage features** with versioned feature store
- 🔄 **Automate deployments** via GitHub Actions CI/CD
- 📦 **Share utilities** through the `january_ml` package

## Quick Start

### Prerequisites
- Python 3.10+
- Snowflake account with appropriate permissions
- Snowflake connection configured locally

### Installation

```bash
# Clone and install dependencies
git clone https://github.com/your-org/snowflake-ml-collaboration.git
cd snowflake-ml-collaboration
pip install -r requirements.txt

# Build shared package
python -m build
```

### Configure Environment

Set these environment variables for local development in the shared sandbox (or use `.env`):

```bash
export SNOWFLAKE_CONNECTION="your_connection_name" # Name of the connection configured in connections.toml. Connection should have role defined.
export SNOWFLAKE_ENVIRONMENT="DEV"  # DEV for local, STAGING/PROD handled by CI/CD
```

**Environment Overview:**
| Environment | Database | Usage |
|-------------|----------|-------|
| `DEV` | `ML_COLLAB_DEV_DB` | Local development & experimentation |
| `STAGING` | `ML_COLLAB_STAGING_DB` | CI/CD deploys from feature branches |
| `PROD` | `ML_COLLAB_PROD_DB` | CI/CD deploys from main branch |

### Implement Feature Views

Add feature generation functions in `feature_store/feature_views.py`:

```python
from snowflake.snowpark import Session, DataFrame

def create_user_features(session: Session) -> DataFrame:
    """Generate user activity features."""
    df = session.table('USER_EVENTS')
    # Feature engineering logic. Must return Snowpark Dataframe.
    return df.select("USER_ID", "CREATED_AT", "FEATURE1", "FEATURE2")
```

### Deploy Feature Store

```bash
python feature_store/setup_feature_store.py
```

Features are automatically versioned based on their definition, enabling reproducibility and lineage tracking. This versioning (using january_ml.utils.version_featureview) allows the pipeline to only re-compute historical data if there have been changes to the definition. If there are no changes, initialization will be skipped.


## Creating ML Projects

### 1. Create a New Project

```bash
./scripts/create_project.sh my_project
cd projects/my_project
```

This creates a project from the template with:
- `config.yml` - DAG configuration
- `utils.py` - Project utilities
- `pip-requirements.txt` - Python dependencies for container notebooks and MLJobs.

### 2. Configure Your DAG

Edit `config.yml` to define your ML workflow:

```yaml
project_name: my_project
active: False # set to True to deploy the project

deploy:
  DAGS:
    - name: TRAINING_PIPELINE
      tasks:
        - name: prepare_data
          file: prepare_data.py      # Python script (direct execution)
          
        - name: train_model
          file: training.py
          mljob: True                # Run as ML Job (containerized)
          dep: prepare_data          # Depends on prepare_data task
          
        - name: evaluate
          file: evaluate.ipynb       # Jupyter notebook
          dep: train_model
          
      schedule: CRON 0 3 * * * UTC   # Daily at 3 AM
      conda_packages:                # Optional additional packages. ONLY necessary for warehouse tasks.
        - xgboost
```

**Task Options:**
- `file`: Python script (`.py`) or Jupyter notebook (`.ipynb`)
- `mljob`: `True` for containerized execution of `.py` files, `False` for warehouse execution (default)
- `dep`: Task name(s) this depends on (creates task graph)
- `final`: `True` for finalizer tasks (run even if dependencies fail)

**Schedule Formats:**
- CRON: `CRON 0 9 * * * UTC` (daily at 9 AM)
- Interval: `3 HOURS`, `30 MINUTES`, `5 SECONDS`
- Empty: Manual trigger only

### 3. Add Your Code

**For Python scripts:**
```python
# prepare_data.py
# Task will execute the main function of the specified script
# first argument must be session
def main(session, **kwargs):
    """Main function for direct execution tasks."""
    # Your ML code here
    df = fs.read_feature_view("FEATURE_VIEW")
    # ... filter and split ...
    return {"data-version":"abc123"} # If returning values from task, must be dict
```

**For ML Jobs:**
```python
# training.py for containerized execution
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version")
    args = parser.parse_args()
    # ... train and log model ...
    return {"model_version":"FANCY_LEMUR_5"} # If returning values from task, must be dict

# MLJob will execute the provided script, must provide main block
if __name__ == "__main__":
    __return__ = main() # must return to __return__ to get output from task
```

**For Snowflake Notebooks:**
```python
# To use january_ml package on Snowflake notebook, run:
from utils import get_stage_packages
get_stage_packages()
! pip install -r pip-requirements.txt
```

### 4. Deploy Your Project

```bash
# Deploy to Snowflake
python scripts/deploy_project.py my_project

# Deploy and run immediately (for testing)
python scripts/deploy_project.py my_project --run-dag
```

This automatically:
- Creates project-specific warehouse and compute pool
- Uploads your code to Snowflake stages
- Deploys notebooks as Snowflake Notebooks
- Creates DAG tasks with dependencies
- Schedules according to your configuration

## Feature Store

### Configure Features

Define entities and feature views in `feature_store/config.yml`:

```yaml
entities:
  - name: user
    join_keys: USER_ID                # column name or list of column names

feature_views:
# provide any optional arguments for FeatureView (https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/feature_store/snowflake.ml.feature_store.FeatureView#snowflake.ml.feature_store.FeatureView)
  - name: user_activity_features
    function: create_user_features    # Function in feature_views.py must return Snowpark dataframe
    entities: user                    # entity or list of entities
    timestamp_col: CREATED_AT
    refresh_freq: 1 day               # Optional auto-refresh
```

## CI/CD Pipeline

The GitHub Actions workflow automatically deploys changes to Snowflake with environment-aware deployments.

### Setup

1. **Add GitHub Secrets** (Settings → Secrets and variables → Actions):
   - `SNOWFLAKE_ACCOUNT` - Your Snowflake account identifier
   - `SNOWFLAKE_USER` - Snowflake service user username
   - `SNOWFLAKE_PASSWORD` - Snowflake user password

2. **Workflow triggers on push** to any branch when files change in:
   - `feature_store/` - Redeploys feature store
   - `projects/` - Redeploys changed projects

### Deployment Flow

The workflow dynamically determines the target environment based on the branch:

```
┌─────────────────────┐
│   Push to Branch    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   GitHub Actions    │
│    • Build pkg      │
│    • Detect changes │
│    • Deploy FS      │
│    • Deploy proj    │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Branch  │ │  main   │
│ != main │ │ branch  │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ STAGING │ │  PROD   │
└─────────┘ └─────────┘
```

**Environment Configuration:**
- **Feature branches** → Deploy to `STAGING` environment
- **Merge to main** → Deploy to `PROD` environment
- **Local development** → Use `DEV` environment (sandbox)

## Repository Structure

```
snowflake-ml-collaboration/
├── .github/workflows/     # CI/CD pipeline
├── feature_store/         # Feature store configuration
│   ├── config.yml
│   ├── feature_views.py
│   └── setup_feature_store.py
├── projects/              # Your ML projects
│   └── my_project/
│       ├── config.yml
│       ├── training.py
│       └── pip-requirements.txt
├── template/              # Project template
├── january_ml/            # Shared utilities package
├── scripts/
│   ├── create_project.sh  # Create new project
│   ├── deploy_project.py  # Deploy to Snowflake
│   └── cleanup.py         # Remove deployed resources
└── dist/                  # Built packages
```

## Best Practices

### Project Organization
- ✅ One project per ML use case
- ✅ Keep projects self-contained
- ✅ Use descriptive task names
- ✅ Test locally before deploying

### Task Dependencies
- ✅ Use `mljob: True` for compute-intensive tasks
- ✅ Use direct execution for lightweight orchestration
- ✅ Pass data via return values or tables
- ✅ Create clear task graphs with meaningful dependencies

### Feature Engineering
- ✅ Centralize features in feature store
- ✅ Use timestamp columns for point-in-time correctness
- ✅ Let automatic versioning handle feature lineage
- ✅ Document feature view purposes

### CI/CD
- ✅ Test changes locally in DEV environment
- ✅ Push to feature branch → auto-deploys to STAGING
- ✅ Merge to main → auto-deploys to PROD
- ✅ Use `--run-dag` flag for smoke tests
- ✅ Review DAG configurations in Snowflake UI
- ✅ Monitor task execution history

## Common Operations

### View Deployed DAGs
```sql
-- In Snowflake
SHOW TASKS IN SCHEMA;
DESCRIBE TASK MY_PROJECT_TRAINING_PIPELINE;
```

### Check Task History
```sql
SELECT * 
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY())
WHERE NAME LIKE 'MY_PROJECT%'
ORDER BY SCHEDULED_TIME DESC;
```

### Update a Project
```bash
# 1. Make changes to your project code/config
# 2. Redeploy
python scripts/deploy_project.py my_project
```

### Delete a Project's Resources

Use the cleanup script to remove all Snowflake resources for a project:

```bash
# Preview what would be deleted (recommended first step)
python scripts/cleanup.py my_project --dry-run

# Actually delete all resources for the project
python scripts/cleanup.py my_project
```

This removes:
- DAGs/Tasks (including all child tasks)
- Notebooks
- Staged files in BUILD_STAGE and JOB_STAGE
- Compute pool
- Warehouse

### Delete Feature Store Resources

```bash
# Delete specific feature views
python scripts/cleanup.py --features user_activity_features --dry-run
python scripts/cleanup.py --features user_activity_features

# Delete multiple feature views
python scripts/cleanup.py --features feature1 feature2 feature3
```

### Delete All Resources

```bash
# Preview full cleanup
python scripts/cleanup.py --all --dry-run

# Delete everything: all projects, all features, and stages
python scripts/cleanup.py --all
```

## Troubleshooting

**Issue: Task fails with import error**
- Ensure `pip-requirements.txt` includes all dependencies
- Rebuild `january_ml`: `python -m build`
- Redeploy project

**Issue: DAG doesn't run on schedule**
- Check task is in STARTED state: `SHOW TASKS`
- Verify schedule format in `config.yml`
- Check warehouse is running

**Issue: Feature store entity not found**
- Register entities before feature views
- Verify entity names match exactly

## Resources

- [Snowflake Tasks & DAGs](https://docs.snowflake.com/en/user-guide/tasks-intro)
- [Snowflake ML Jobs](https://docs.snowflake.com/en/developer-guide/snowpark-ml/ml-jobs)
- [Snowflake Feature Store](https://docs.snowflake.com/en/developer-guide/snowpark-ml/feature-store/overview)

---

**Questions?** Check task history in Snowflake or review deployment logs in GitHub Actions.
