import pandas as pd

from core_ml.common.queries import _fetch_list_all_available_asset_ids


def _parse_asset_ids(asset_id: str) -> list[int] | str:
    """Parse asset_id input to a list of asset ids."""
    asset_id = asset_id.strip()

    # Handle None or empty
    if not asset_id or asset_id.lower() in ("all", "none"):
        return "all"

    # Handle "[1,2,3]" format - strip brackets
    if asset_id.startswith("[") and asset_id.endswith("]"):
        asset_id = asset_id[1:-1].strip()

    # Handle "1,2,3" format (comma-separated)
    if "," in asset_id:
        try:
            return [int(x.strip()) for x in asset_id.split(",")]
        except (ValueError, TypeError):
            raise ValueError(f"Invalid asset_id format: '{asset_id}'. Expected comma-separated integers.")

    # Handle single integer as string
    try:
        return [int(asset_id)]
    except (ValueError, TypeError):
        raise ValueError(f"Invalid asset_id: '{asset_id}'. Must be int(s) or comma-separated string.")


def _filter_invalid_asset_ids(list_asset_ids: list[int] | str, list_all_available_asset_ids: list[int]) -> list[int]:
    """Filter out invalid asset ids from the input list."""
    if list_asset_ids == "all":
        return list_all_available_asset_ids

    list_valid_asset_ids = [id for id in list_asset_ids if id in list_all_available_asset_ids]
    return list_valid_asset_ids


def select_only_valid_asset_ids(asset_id: str = "all") -> list[int]:
    """Select list of valid asset ids based on input."""
    list_asset_ids = _parse_asset_ids(asset_id)

    list_all_available_asset_ids = _fetch_list_all_available_asset_ids()

    list_valid_asset_ids = _filter_invalid_asset_ids(list_asset_ids, list_all_available_asset_ids)
    return list_valid_asset_ids


def convert_input_from_df_to_dict(df: pd.DataFrame, list_asset_ids: list[int]) -> dict[int, pd.DataFrame]:
    """one df with all assets -> dict of one df per asset_id"""
    return {asset_id: df[df["asset_id"] == asset_id].drop(columns=["asset_id"]) for asset_id in list_asset_ids}
