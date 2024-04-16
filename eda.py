from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
from typing import List
import numpy as np

@dataclass
class StructTypeInfo:
    datatypes: List[str]
    frequency: List[int]
    columns: List[List[str]]

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "datatypes": self.datatypes,
            "frequency": self.frequency,
            "columns": self.columns,
        }

        return pd.DataFrame(data)

class Eda(ABC):
    @abstractmethod
    def structtype_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """The struct type of the dataframe"""

    @abstractmethod
    def completness_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """the completness of the dataframe"""

    @abstractmethod
    def highcardinality_info(self, df: pd.DataFrame) -> List[dict]:
        """the completness of the dataframe"""

    def structure_validation(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
        """Validation of structure"""

    @abstractmethod
    def feature_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Information about object features in the dataset"""

    @abstractmethod
    def missinginformation_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Information about missing values"""

    @abstractmethod
    def duplicated_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Information about duplicated rows"""

class PreprocessEda(Eda):
    def __init__(self):
        self.structinfo = None

    def structtype_info(self, df: pd.DataFrame) -> StructTypeInfo:
        datatypes = pd.value_counts(df.dtypes).keys().tolist()
        frequency = pd.value_counts(df.dtypes).values.tolist()
        columns = [df.select_dtypes(include=key).columns.tolist() for key in pd.value_counts(df.dtypes).keys()]
        self.structinfo = StructTypeInfo(datatypes, frequency, columns)
        return self.structinfo
    
    def completness_info(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df.isnull().sum()/df.shape[0]).sort_values(ascending=False)
    
    def highcardinality_info(self, df: pd.DataFrame) -> List[dict]:
        result = [
            {
                "column": column,
                "nunique %": df.loc[~df[column].isnull(), column].nunique() / df.loc[~df[column].isnull(), column].shape[0]
            }
            if df.loc[~df[column].isnull(), column].nunique() != 0
            else {"column": column, "nunique %": 1.00}
            for column in df.columns
        ]

        result = sorted(result, key=lambda x: x["nunique %"], reverse=True)

        return result
    
    def structure_validation(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
        if self.structinfo is None:
            struct = self.structtype_info(df)
            struct = struct.to_dataframe()
        else:
            struct = self.structinfo.to_dataframe()

        if type == "object":
            struct = df[struct.loc[struct["datatypes"] == "object"]["columns"].values.tolist()[0]]
        elif type == "int":
            struct = df[struct.loc[struct["datatypes"] == "int"]["columns"].values.tolist()[0]]
        elif type == "float": 
            struct = df[struct.loc[struct["datatypes"] == "float"]["columns"].values.tolist()[0]]
        else:
            raise ValueError("type is not defined")
        
        return struct
    
    def feature_info(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
        struct = self.structure_validation(df, type)

        object = pd.DataFrame({
            "Features": struct.nunique().sort_values(ascending=False).keys(),
            "Frequency": struct.nunique().sort_values(ascending=False).values
        })
        
        return object
    
    def missinginformation_info(self, df:pd.DataFrame, type: str) -> pd.DataFrame:
        struct = self.structure_validation(df, type)
        
        object = pd.DataFrame({
            "Features": struct.isnull().sum().sort_values(ascending=False).keys(),
            "Missing": np.round(struct.isnull().sum().sort_values(ascending=False).values / struct.shape[0], 2)
        })

        return object
    
    def duplicated_info(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
        struct = self.structure_validation(df, type)

        object = pd.DataFrame({
            "Features": struct.isnull().sum().sort_values(ascending=False).keys(),
            "Duplicated": np.round(struct.isnull().sum().sort_values(ascending=False).values / struct.shape[0], 2)
        })

        return object