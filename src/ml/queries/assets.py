from toolkit.data.query import Query
from toolkit.database import database


def _fetch_list_all_available_asset_ids(list_asset_types: list[str] = ["Wind farm", "Solar farm"]) -> list[int]:
    """Get list of all available asset ids."""
    sql_query = """
    SELECT
        asset.id AS asset_id
    FROM
        data_lake.asset AS asset
    INNER JOIN
        data_lake.asset_type AS asset_type
    	ON asset.asset_type_id = asset_type.id
    ORDER BY 1
    """
    query = Query(sql_query)
    query.with_in("asset_type.name", list_asset_types)
    with database.session() as session:
        result = session.execute(query.prepared_statement).mappings().fetchall()

    list_asset_ids = [row["asset_id"] for row in result]
    return list_asset_ids
