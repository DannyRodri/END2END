import pandera.pandas as pa
from pandera import Column, DataFrameSchema
import pandas as pd
from pandera.errors import SchemaErrors

def build_schema(structure: dict) -> DataFrameSchema:
    type_map = {
        "int64": pa.Int64,
        "float64": pa.Float64,
        "object": pa.String
    }

    schema_columns = {}

    for col, dtype in structure["columnas"].items():
        schema_columns[col] = Column(
            type_map[dtype],
            nullable=True  # Permite valores nulos, ajusta según tus necesidades
        )

    return DataFrameSchema(
        schema_columns,
        strict=True   # Impide columnas adicionales no definidas en el esquema
    )


