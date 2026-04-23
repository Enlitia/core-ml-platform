import pandas as pd
from sqlalchemy import text
from toolkit.database import database


class DBGateway:
    def __init__(self) -> None:
        pass

    @staticmethod
    def fetch_df(query: str) -> pd.DataFrame:
        """Execute a query and return a DataFrame."""
        with database.session() as session:
            result = session.execute(text(query))
            data = result.fetchall()
            columns = result.keys()
            df = pd.DataFrame(data, columns=columns)
        return df

    @staticmethod
    def insert_update_df(
        df: pd.DataFrame,
        table_name: str,
        update_columns: list[str] | None = None,
        conflict_columns: list[str] | None = None,
    ) -> None:
        """Insert or update DataFrame data on conflict.

        Args:
            df: DataFrame to save
            table_name: Target table name
            update_columns: Columns to update on conflict (None = update all columns)
            conflict_columns: Columns that define conflicts (e.g., ['asset_id', 'read_at'])
        """
        if df.empty:
            return

        # Replace NaN with None for SQL compatibility
        df_obj = df.astype(object)
        null_mask: pd.DataFrame = pd.notnull(df_obj)
        df_clean = df_obj.where(null_mask, None)  # type: ignore[call-overload]

        columns = list(df_clean.columns)
        update_cols = update_columns if update_columns else columns

        with database.session() as session:
            # Build INSERT statement
            cols_str = ", ".join(columns)
            placeholders = ", ".join([f":{col}" for col in columns])

            # Build ON DUPLICATE KEY UPDATE clause (MySQL/MariaDB)
            # Or ON CONFLICT DO UPDATE (PostgreSQL)
            update_str = ", ".join([f"{col} = VALUES({col})" for col in update_cols])

            # Try PostgreSQL syntax first, fallback to MySQL
            if conflict_columns:
                # PostgreSQL: ON CONFLICT (columns) DO UPDATE SET ...
                conflict_str = ", ".join(conflict_columns)
                update_pairs = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
                query = f"""
                    INSERT INTO {table_name} ({cols_str})
                    VALUES ({placeholders})
                    ON CONFLICT ({conflict_str}) DO UPDATE SET {update_pairs}
                """
            else:
                # MySQL/MariaDB: ON DUPLICATE KEY UPDATE
                query = f"""
                    INSERT INTO {table_name} ({cols_str})
                    VALUES ({placeholders})
                    ON DUPLICATE KEY UPDATE {update_str}
                """

            # Execute for each row
            for _, row in df_clean.iterrows():
                session.execute(text(query), row.to_dict())

            session.commit()
