# Documentation Workflow

- Edit Markdown sources in this folder (`index.md`, `users.md`, `developers.md`, `api.md`, `configuration.md`, `deployment.md`).
- Preview changes live with:
  ```bash
  pip install -r ../requirements-docs.txt
  mkdocs serve
  ```
- Build static HTML into `site/` for GitHub Pages or any static host:
  ```bash
  mkdocs build --clean
  ```
- Keep navigation in sync by updating `../mkdocs.yml` whenever you add or rename pages.
