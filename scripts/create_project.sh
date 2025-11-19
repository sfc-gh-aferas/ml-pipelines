#!/bin/bash

# Script to create a new ML project from template
# Usage: ./create_project.sh <project_name>

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if project name is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Project name is required${NC}"
    echo "Usage: $0 <project_name>"
    exit 1
fi

PROJECT_NAME="$1"

# Get the repository root directory (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TEMPLATE_DIR="$REPO_ROOT/template"
PROJECTS_DIR="$REPO_ROOT/projects"
TARGET_DIR="$PROJECTS_DIR/$PROJECT_NAME"

# Validate template directory exists
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo -e "${RED}Error: Template directory not found at $TEMPLATE_DIR${NC}"
    exit 1
fi

# Check if project already exists
if [ -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error: Project '$PROJECT_NAME' already exists at $TARGET_DIR${NC}"
    exit 1
fi

# Create projects directory if it doesn't exist
mkdir -p "$PROJECTS_DIR"

echo -e "${YELLOW}Creating new project: $PROJECT_NAME${NC}"
echo "Template: $TEMPLATE_DIR"
echo "Target:   $TARGET_DIR"
echo ""

# Copy template directory to new project location
cp -r "$TEMPLATE_DIR" "$TARGET_DIR"
echo -e "${GREEN}✓${NC} Copied template files"

# Update utils.py to replace template with project_name in BUILD_STAGE path
if [ -f "$TARGET_DIR/utils.py" ]; then
    # Use sed to replace @BUILD_STAGE/template/dist with @BUILD_STAGE/{project_name}/dist
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed syntax
        sed -i '' "s|@BUILD_STAGE/template/dist|@BUILD_STAGE/$PROJECT_NAME/dist|g" "$TARGET_DIR/utils.py"
    else
        # Linux sed syntax
        sed -i "s|@BUILD_STAGE/template/dist|@BUILD_STAGE/$PROJECT_NAME/dist|g" "$TARGET_DIR/utils.py"
    fi
    echo -e "${GREEN}✓${NC} Updated utils.py with project-specific stage path"
else
    echo -e "${YELLOW}⚠${NC}  utils.py not found, skipping stage path update"
fi

# Update config.yml to replace <PROJECT_NAME> with actual project name
if [ -f "$TARGET_DIR/config.yml" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed syntax
        sed -i '' "s/<PROJECT_NAME>/$PROJECT_NAME/g" "$TARGET_DIR/config.yml"
    else
        # Linux sed syntax
        sed -i "s/<PROJECT_NAME>/$PROJECT_NAME/g" "$TARGET_DIR/config.yml"
    fi
    echo -e "${GREEN}✓${NC} Updated config.yml with project name"
else
    echo -e "${YELLOW}⚠${NC}  config.yml not found, skipping project name update"
fi

echo ""
echo -e "${GREEN}✅ Project '$PROJECT_NAME' created successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Navigate to your project: cd projects/$PROJECT_NAME"
echo "  2. Edit config.yml to configure your DAGs"
echo "  3. Add your Python scripts or Jupyter notebooks"
echo "  4. Update pip-requirements.txt with your dependencies"
echo "  5. Deploy your project: python scripts/deploy_project.py $PROJECT_NAME"
echo ""

