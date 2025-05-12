This document intended to reflect the architecture of the system for further development.

# Techstack
1. Wandb (Model hyperparameter tuning)
2. Argparse (Intended purpose)
3. Poetry (Dependency management)
    - Because uv initial folder is complex although faster
    - Conda is slower, consider pip on WSL

# Folder Architecture

1. /[\MAIN_MODULE] (Module folder)
2. /hyperparameters (Wandb single config and hyperparameter sweep)
3. /data (Data folder)
4. /scripts (Scripts folder)

Optional Folder
5. /logs (Logs folder)
6. /ref (Reference folder)
7. /tests (Python tests folder)
8. /docs (Documentation folder or results)
9. /examples (Example folder)