# Migrating hadisakho/cacib-hicp to v2

These steps replace the old code with the simplified version on a new branch, so nothing is lost. Commands are for macOS (Terminal or the PyCharm terminal), run from the root of your `cacib-hicp` clone. Delete this file once you're done.

## 1. Save the current version
```bash
git checkout main && git pull
git status                       # must be clean; commit or stash anything pending
git tag v1-original              # the old version stays reachable with: git checkout v1-original
git push origin v1-original
git checkout -b simplify
```

## 2. Move local data out of the repo
The old `data/` and `reports/` folders aren't tracked by git, but their layout differs from v2. Move them aside so they don't get committed by mistake.
```bash
mkdir -p ../cacib-hicp-old-local
[ -d data ] && mv data ../cacib-hicp-old-local/
[ -d reports ] && mv reports ../cacib-hicp-old-local/
[ -d .venv ] && rm -rf .venv     # recreated by uv in step 4
```

## 3. Remove the old files
```bash
git rm -r -q cacib_lab example_results deploy notebooks .idea
git rm -q *.py METHODOLOGY.md VERIFICATION.md GUIDE_COMPLETION.md MANIFEST.sha256 \
          Dockerfile render.yaml requirements-lab.txt README.md pyproject.toml uv.lock
rm -rf cacib_lab deploy notebooks __pycache__ .pytest_cache .ruff_cache .ipynb_checkpoints   # untracked leftovers
ls -A                            # should now list only .git (and maybe .DS_Store)
```

## 4. Copy in the new files
Unzip `cacib-hicp-v2.zip` (to `~/Downloads/cacib-hicp-v2`), then:
```bash
rsync -a ~/Downloads/cacib-hicp-v2/ ./
uv sync --locked                 # creates .venv with pinned versions
uv run pytest                    # expect: 27 passed
```

## 5. Re-run the pipeline on the Eurostat API
The `data/processed/` and `outputs/` folders in the zip were already built from the Eurostat API (data to September 2026). Re-running confirms everything works on your machine:
```bash
uv run python -m hicp.pipeline --refresh
```
Expected output:
- `checks: 18/18 passed`;
- `4-model average (recommended) RMSE 0.664 pp, 0.77 x no change`;
- forecast `2026-10 3.96%, 2026-11 4.05%, 2026-12 4.14%`.

After each new Eurostat release (mid-month), the same command updates everything.

## 6. Check the dashboard and the notebook
```bash
uv run streamlit run app.py                          # http://localhost:8501
uv run jupyter lab notebooks/hicp_analysis.ipynb     # Run All
```
In PyCharm, set the interpreter to the project's `.venv`: *Settings → Python Interpreter → Add Interpreter → Add Local Interpreter → Select existing → `.venv/bin/python`*.

## 7. Commit, push, merge
```bash
git add -A
git status                       # check that no large or unexpected files are staged
git commit -m "Simplify pipeline; 4-model average forecast; shock pass-through; Streamlit dashboard"
git push -u origin simplify
```
Open the pull request on GitHub (`simplify` → `main`), review it, and merge.

## 8. Host the dashboard (Streamlit Community Cloud, free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click *Create app*, choose repository `hadisakho/cacib-hicp`, branch `main`, main file `app.py`.
3. Under *Advanced settings*, choose Python 3.12, then deploy. Put the app URL at the top of the README.

## 9. Turn on the monthly refresh (optional)
On GitHub: *Settings → Actions → General → Workflow permissions → Read and write permissions*. The workflow `.github/workflows/refresh.yml` then runs on the 20th of each month; you can also run it by hand from the *Actions* tab.
