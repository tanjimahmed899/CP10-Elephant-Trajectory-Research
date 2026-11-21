# GitHub setup from Windows + Ubuntu WSL

1. Extract `CP10_GitHub_Research_Repository.zip` somewhere in your Ubuntu home, not under a system directory.
2. Enter `CP10-Elephant-Trajectory-Research/` and run `python3 scripts/pre_push_audit.py`.
3. Review `README.md`, `LICENSE-STATUS.md`, `data/README.md`, manifests and any paths/metadata for publication rights. Keep **private** initially.
4. If available, sign in with `gh auth login` (choose GitHub.com; browser authentication). Never paste your token or password into a chat.
5. Run `bash scripts/publish_private_with_gh.sh` to create a new private GitHub repository and push; the script asks for confirmation before any network upload.
6. If the `gh` CLI is absent, create a **private EMPTY** repository at GitHub, then use `git init -b main; git add .; git commit -m 'Initial research archive'; git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git; git push -u origin main` after verifying `git status`.

**Additional E: drive scripts/results not mounted here:** `python3 scripts/collect_additional_wsl.py` previews them; adding `--apply` makes a copy under `additional_wsl/`. Review contents, run the pre-push audit, commit and push explicitly. The `--include-results` flag also copies small CSV/JSON/Markdown/TXT results, excluding known sensitive camera and location file names.

**Actual raw datasets, camera metadata and models are NOT bundled:** source archives and all model assets remain on your Windows/WSL project. Use controlled private storage and SHA checks, not casual public Git commits. GitHub file size limit applies to each file, and LFS has storage/bandwidth quotas.
