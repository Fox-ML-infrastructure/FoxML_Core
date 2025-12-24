# UPDATE Directory

Tracks updates and changes made to the codebase, organized by date and timestamp.

## Structure

```
UPDATE/
  YYYY-MM-DD/          # Date folders
    HH-MM-SS/          # Timestamp folders (when changes were made)
      CHANGES.md       # Description of changes
      files_changed.txt # List of files modified
```

## Usage

### Option 1: Python Script

```bash
# Quick entry with description
python SCRIPTS/utils/create_update_entry.py "Fixed bug in target ranking"

# Interactive mode (prompts for description, opens editor)
python SCRIPTS/utils/create_update_entry.py --interactive
```

### Option 2: Shell Script

```bash
# Quick entry
./SCRIPTS/utils/create_update_entry.sh "Fixed bug in target ranking"

# Interactive (prompts for description)
./SCRIPTS/utils/create_update_entry.sh
```

### Option 3: Manual

```bash
# Create directory
mkdir -p UPDATE/$(date +%Y-%m-%d)/$(date +%H-%M-%S)

# Create CHANGES.md and files_changed.txt manually
```

## Example

```
UPDATE/
  2024-12-19/
    14-30-00/
      CHANGES.md
      files_changed.txt
    15-45-30/
      CHANGES.md
      files_changed.txt
  2024-12-20/
    09-15-00/
      CHANGES.md
      files_changed.txt
```

## CHANGES.md Template

Each entry should include:
- Summary: Brief description
- Changes Made: Detailed list of changes
- Files Modified: List of affected files
- Impact: What this changes affects
- Testing: Testing performed or needed

## Best Practices

1. Create entry before making changes - Helps track what you're about to do
2. Update after changes - Fill in details of what was actually changed
3. Include all modified files - The script auto-detects from git, but verify
4. Be descriptive - Future you will thank present you
5. Link related entries - Reference previous UPDATE entries if building on them

## Integration with Git

Helper scripts automatically detect modified files from git:
- Modified files: `git diff --name-only HEAD`
- New files: `git ls-files --others --exclude-standard`

If not in a git repo, manually list files in `files_changed.txt`.
