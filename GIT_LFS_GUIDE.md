# Git Large File Storage (LFS) Guide

## Overview

This repository uses Git LFS to handle genuinely large binary files efficiently.

**Important**: Only binary files that cannot be regenerated are tracked via LFS.
Text files, JSON files, databases, and logs should NOT be tracked via LFS.

## Current LFS Configuration

### Tracked File Types (`.gitattributes`)

| Pattern | Description |
|---------|-------------|
| `*.dll`, `*.so`, `*.dylib` | Native binary libraries |
| `*.model`, `*.h5`, `*.pkl`, `*.onnx` | AI/ML model files |
| `*.zip`, `*.tar.gz`, `*.rar` | Archive files |
| `*.mp4`, `*.mov`, `*.avi` | Video files |
| `*.wav`, `*.mp3` | Audio files |
| `*.psd`, `*.tiff`, `*.tif` | Large image formats |
| `*.parquet` | Large data format |

### NOT Tracked via LFS (by design)

| Pattern | Reason |
|---------|--------|
| `*.json` | Breaks `composer.json`, `package.json`, schema files |
| `*.csv` | Usually small enough for regular git |
| `*.db`, `*.sqlite` | Runtime-generated, should be in `.gitignore` |
| `*.log` | Runtime-generated, should be in `.gitignore` |
| `*.bin` | Too broad, catches non-binary files |
| `*.txt`, `*.md` | Text files belong in regular git |

## Usage Instructions

### Cloning the Repository

```bash
# Standard clone (LFS files download automatically)
git clone https://github.com/skintwin-ai/ojskin.git

# If LFS is not installed, clone without LFS:
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/skintwin-ai/ojskin.git
```

### After Cloning

```bash
# Install PHP dependencies (regenerates vendor/ directories)
composer --working-dir=lib/pkp install --no-dev
composer --working-dir=plugins/paymethod/paypal install
composer --working-dir=plugins/generic/citationStyleLanguage install

# Install Node.js dependencies (regenerates node_modules/)
cd skz-integration/simulation-dashboard && npm install
cd ../workflow-visualization-dashboard && npm install
```

## Troubleshooting

### LFS Objects Not Found (404)

If you see "Object does not exist on the server" errors:
1. The LFS objects may not have been uploaded yet
2. Use `GIT_LFS_SKIP_SMUDGE=1` when cloning to skip LFS downloads
3. After cloning, run `git lfs pull` to retry downloading LFS objects

### Adding New Large Files

1. Ensure the file type is already tracked: `git lfs track`
2. If not, add the pattern: `git lfs track "*.newextension"`
3. Commit `.gitattributes` first, then commit the large file
4. Push with `git push` (LFS objects upload automatically)
