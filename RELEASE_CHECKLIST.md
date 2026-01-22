# Release Checklist for HyMLX 0.2.0

## 1. Final Verification
- [x] **Build**: `uv build` succeeded (Artifacts in `dist/`).
- [x] **Tests**: Core tests passed (43/43).
- [x] **Metadata**: `pyproject.toml` version is 0.2.0.
- [x] **Documentation**: `README.md` and `LICENSE` are present.
- [x] **Cleanliness**: `.gitignore` created.

## 2. GitHub Publication
Since `hymlx` is now in `~/i/hymlx`, you can push it to a new or existing repository.

### Option A: Push to Soft-Machine-io/soft-machine (Monorepo/Folder)
If this is part of the existing `soft-machine` repo:
```bash
cd ~/i/hymlx
git init  # If not already initialized
git remote add origin https://github.com/Soft-Machine-io/soft-machine.git
git add .
git commit -m "feat(hymlx): Release 0.2.0"
git push -u origin main
```

### Option B: New Repository
1. Create a new empty repo on GitHub named `hymlx`.
2. Push:
```bash
cd ~/i/hymlx
git init
git add .
git commit -m "Initial release 0.2.0"
git branch -M main
git remote add origin https://github.com/Soft-Machine-io/hymlx.git
git push -u origin main
```

## 3. PyPI Publication
You can publish the built artifacts in `dist/` to PyPI.

### Using Twine (Standard)
```bash
# Install twine if needed
pip install twine

# Upload to TestPyPI first (Optional but recommended)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### Using UV (Experimental)
If you have `uv` configured with PyPI credentials:
```bash
uv publish
```

## 4. Post-Release
- [ ] Create a Release tag on GitHub (v0.2.0).
- [ ] Verify installation: `pip install hymlx`.
