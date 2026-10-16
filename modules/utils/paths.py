import os
import platform


def get_user_data_dir(app_name: str = "ComicTranslate") -> str:
    """
    Returns the platform-specific user data directory for the application.
    
    Windows: %LOCALAPPDATA%/<app_name>
    macOS: ~/Library/Application Support/<app_name>
    Linux: $XDG_DATA_HOME/<app_name> or ~/.local/share/<app_name>
    """
    system = platform.system()
    
    if system == "Windows":
        base_dir = os.getenv('LOCALAPPDATA')
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif system == "Darwin":
        base_dir = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        # Linux / Unix
        base_dir = os.getenv('XDG_DATA_HOME')
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".local", "share")
            
    return os.path.join(base_dir, app_name)


def get_default_project_autosave_dir(folder_name: str = "Comic Translate") -> str:
    """
    Returns a user-facing default folder for project auto-save files.

    Windows/macOS/Linux: ~/Documents/<folder_name>
    """
    return os.path.join(os.path.expanduser("~"), "Documents", folder_name)


def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_llm_error_log_path(filename: str = "llm_translation_errors.log") -> str:
    """Returns the path to the LLM error log file, creating the directory if needed."""
    try:
        project_root = get_project_root()
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # Fallback to user data dir or temp dir if project root is not writable
        try:
            log_dir = os.path.join(get_user_data_dir(), "logs")
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            import tempfile
            log_dir = tempfile.gettempdir()
    return os.path.join(log_dir, filename)
