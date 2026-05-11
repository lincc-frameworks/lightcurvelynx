from pathlib import Path
from unittest.mock import patch

from lightcurvelynx import _get_download_data_dir
from lightcurvelynx.utils.data_download import download_data_file_if_needed


def test_get_download_data_dir_default(monkeypatch):
    """Test that the default download dir follows XDG spec (~/.cache/lightcurvelynx)."""
    monkeypatch.delenv("LIGHTCURVELYNX_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert _get_download_data_dir() == Path.home() / ".cache" / "lightcurvelynx"


def test_get_download_data_dir_xdg(monkeypatch, tmp_path):
    """Test that XDG_CACHE_HOME is respected."""
    monkeypatch.delenv("LIGHTCURVELYNX_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _get_download_data_dir() == tmp_path / "lightcurvelynx"


def test_get_download_data_dir_package_override(monkeypatch, tmp_path):
    """Test that LIGHTCURVELYNX_DATA_DIR takes precedence over XDG_CACHE_HOME."""
    monkeypatch.setenv("LIGHTCURVELYNX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _get_download_data_dir() == tmp_path


def test_download_data_file_if_needed(tmp_path, capsys):
    """Test the functionality of downloading a data file using pooch."""
    data_url = "mock"

    def mock_urlretrieve(url, known_hash, fname, path):
        full_name = path / fname
        with open(full_name, "w") as f:
            f.write("Mock data")
        return full_name

    with patch("pooch.retrieve", side_effect=mock_urlretrieve):
        # Create an existing data file.
        data_path_1 = tmp_path / "test_data.dat"
        with open(data_path_1, "w") as f:
            f.write("Test")
        assert data_path_1.exists()

        # If we try to download the file again, it should not overwrite it.
        assert download_data_file_if_needed(data_path_1, data_url, force_download=False)
        with open(data_path_1, "r") as f:
            assert f.read() == "Test"

        # If we force the download, it should overwrite the existing file.
        assert download_data_file_if_needed(data_path_1, data_url, force_download=True, silent=True)
        with open(data_path_1, "r") as f:
            assert f.read() == "Mock data"

        # With silent=True, we not should see print statements from the download process.
        captured = capsys.readouterr()
        assert "Downloading data file from" not in captured.out

        # Create a second data file that does not exist.
        data_path_2 = tmp_path / "test_data_2.dat"
        assert not data_path_2.exists()

        # Download the second file without forcing it.
        assert download_data_file_if_needed(data_path_2, data_url, force_download=False)
        with open(data_path_2, "r") as f:
            assert f.read() == "Mock data"

        # With silent=False, we should see print statements from the download process.
        captured = capsys.readouterr()
        assert "Downloading data file from" in captured.out
