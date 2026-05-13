"""Allow ``python -m src script.txt video.mp4`` as the primary entrypoint."""

from .cli import main


if __name__ == "__main__":
    main()
