from pathlib import Path
from kedro.io import DataCatalog
from kedro.config import OmegaConfigLoader


def load_catalog(
    conf_dir: str | Path = "conf",
    env: str | None = None,
    credentials: dict | None = None,
) -> tuple[DataCatalog, dict]:
    """
    Carga Kedro DataCatalog y parámetros de forma reusable.

    Args:
        conf_dir: ruta a conf/
        env: environment (local, dev, prod)
        credentials: credenciales opcionales

    Returns:
        catalog: Kedro DataCatalog
        params: diccionario de parámetros
    """
    conf_path = Path(conf_dir)

    config_loader = OmegaConfigLoader(
        conf_source=conf_path,
        env=env
    )

    catalog = DataCatalog.from_config(
        catalog=config_loader["catalog"],
        credentials=credentials or {}
    )

    params = config_loader["parameters"]

    return catalog, params
