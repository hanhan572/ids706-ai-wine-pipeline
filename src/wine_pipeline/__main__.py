"""Allow ``python -m wine_pipeline`` to run the command-line interface."""

from wine_pipeline.cli import main

raise SystemExit(main())
