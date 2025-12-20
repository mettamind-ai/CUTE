#!/bin/bash

# Color setup following SCRIPTS.md conventions
ESC=$'\033'
RED="${ESC}[31m"
GREEN="${ESC}[32m"
YELLOW="${ESC}[33m"
CYAN="${ESC}[36m"
BOLD="${ESC}[1m"
DIM="${ESC}[2m"
RESET="${ESC}[0m"

# Auto-disable colors when not supported or NO_COLOR is set
if [[ -n "${NO_COLOR:-}" ]] || ! [[ -t 1 ]] || ! command -v tput >/dev/null 2>&1 || [[ $(tput colors 2>/dev/null || echo 0) -lt 8 ]]; then
    ESC="" RED="" GREEN="" YELLOW="" CYAN="" BOLD="" DIM="" RESET=""
fi

# Track statistics using temp file (to avoid subshell variable scope issues)
STATS_FILE=$(mktemp)
SKIPPED_FILE=$(mktemp)
trap "rm -f $STATS_FILE $SKIPPED_FILE" EXIT
echo "0 0 0 0" > "$STATS_FILE"
: > "$SKIPPED_FILE"

echo -e "${BOLD}${CYAN}Updating all Git repositories...${RESET}"
echo ""

# Function to update a repository
update_repo() {
    local repodir="$1"
    local reponame=$(basename "$repodir")
    local needs_rename_back=false
    
    # Read current stats
    read -r total updated failed skipped < "$STATS_FILE"
    
    # Check if _git exists and needs to be renamed
    if [[ -d "$repodir/_git" ]] && [[ ! -d "$repodir/.git" ]]; then
        echo -e "${YELLOW}Found _git in ${CYAN}$reponame${YELLOW}, temporarily renaming to .git...${RESET}"
        if mv "$repodir/_git" "$repodir/.git" 2>/dev/null; then
            echo -e "${GREEN}✓${RESET} Renamed _git to .git"
            needs_rename_back=true
        else
            echo -e "${RED}✗ Failed to rename _git${RESET}"
            echo "$total $updated $((failed + 1)) $skipped" > "$STATS_FILE"
            return
        fi
    fi
    
    # Change to the repository directory
    if ! cd "$repodir" 2>/dev/null; then
        echo -e "${RED}${BOLD}Error:${RESET} Cannot access ${CYAN}$repodir${RESET}"
        echo "$total $updated $((failed + 1)) $skipped" > "$STATS_FILE"
        return
    fi
    
    total=$((total + 1))
    
    # Print current directory
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}Repository:${RESET} ${CYAN}$reponame${RESET} ${DIM}($(pwd))${RESET}"
    
    # Check if it's a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning:${RESET} Not a valid git repository"
        echo "$reponame (not a git repo)" >> "$SKIPPED_FILE"
        echo "$total $updated $failed $((skipped + 1))" > "$STATS_FILE"
        echo ""
        cd - > /dev/null
        
        # Rename back if needed
        if [[ "$needs_rename_back" == "true" ]]; then
            mv "$repodir/.git" "$repodir/_git" 2>/dev/null
        fi
        return
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo -e "${YELLOW}Warning:${RESET} Uncommitted changes detected, skipping pull"
        echo "$reponame (uncommitted changes)" >> "$SKIPPED_FILE"
        echo "$total $updated $failed $((skipped + 1))" > "$STATS_FILE"
        echo ""
        cd - > /dev/null
        
        # Rename back if needed
        if [[ "$needs_rename_back" == "true" ]]; then
            mv "$repodir/.git" "$repodir/_git" 2>/dev/null
            echo -e "${DIM}Renamed .git back to _git${RESET}"
            echo ""
        fi
        return
    fi
    
    # Run git pull
    if git pull 2>&1; then
        echo -e "${GREEN}✓${RESET} Updated successfully"
        echo "$total $((updated + 1)) $failed $skipped" > "$STATS_FILE"
    else
        echo -e "${RED}${BOLD}✗ Failed to update${RESET}"
        echo "$total $updated $((failed + 1)) $skipped" > "$STATS_FILE"
    fi
    
    # Rename back if needed
    if [[ "$needs_rename_back" == "true" ]]; then
        cd - > /dev/null
        if mv "$repodir/.git" "$repodir/_git" 2>/dev/null; then
            echo -e "${DIM}Renamed .git back to _git${RESET}"
        else
            echo -e "${RED}✗ Failed to rename .git back to _git${RESET}"
        fi
    else
        cd - > /dev/null
    fi
    
    echo ""
}

# Find all directories containing .git or _git
{
    find . -name ".git" -type d
    find . -name "_git" -type d
} | sort -u | while IFS= read -r gitdir; do
    # Get the parent directory (the actual repo directory)
    repodir=$(dirname "$gitdir")
    update_repo "$repodir"
done

# Read final stats
read -r total updated failed skipped < "$STATS_FILE"

# Print summary
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}Summary:${RESET}"
echo -e "  Total repositories: ${BOLD}$total${RESET}"
echo -e "  ${GREEN}Updated: $updated${RESET}"
if [[ $skipped -gt 0 ]]; then
    echo -e "  ${YELLOW}Skipped: $skipped${RESET}"
    while IFS= read -r skipped_repo; do
        [[ -z "$skipped_repo" ]] && continue
        echo -e "    ${DIM}- $skipped_repo${RESET}"
    done < "$SKIPPED_FILE"
fi
if [[ $failed -gt 0 ]]; then
    echo -e "  ${RED}Failed: $failed${RESET}"
fi
