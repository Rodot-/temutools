import re
import io
import sys
import yaml
from typing import Dict, List, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import pandas as pd
from astropy import units as u


DEFAULT_UNITS = {"velocity": "km/s", "density": "g/cm^3", "t_rad": "K"}


@dataclass
class CSVYField:
    """Header info for the CSV Part of the CSVY File"""

    name: str
    unit: Union[str, u.Quantity, None] = None
    desc: Union[str, None] = None

    def __post_init__(self):
        if self.unit is None and self.name in DEFAULT_UNITS:
            self.unit = DEFAULT_UNITS[self.name]
        if self.desc is None:  # Make a nice default description
            if re.match("[A-Z][a-z]?\d*?", self.name) is not None:
                self.desc = f"Mass fraction of {self.name} in each shell"
            elif self.name == "velocity":
                self.desc = "Velocity at the outer boundary of each shell"
            elif self.name == "density":
                self.desc = "Density in each shell"
            elif self.name == "dilution_factor":
                self.desc = "Dilution Factor in each shell"
            elif self.name == "t_rad":
                self.desc = "Radiative Temperature in each shell"


@dataclass
class CSVYHeader:
    """Header info for the YAML part of the CSVY File"""

    model_density_time_0: u.Quantity
    model_isotope_time_0: u.Quantity

    v_inner_boundary: Union[u.Quantity, None] = None
    v_outer_boundary: Union[u.Quantity, None] = None

    name: str = None
    description: str = None

    tardis_model_config_version: str = "v1.0"

    datatype: Dict[str, List[CSVYField]] = field(
        default_factory=lambda: {"fields": list()}
    )


class CSVYWriter:
    """Convienience Class for Writing out csvy files"""

    # Valid fields other than abundances
    valid_fields = ["velocity", "density", "t_rad", "dilution_factor"]

    def __init__(
        self,
        csv_data: pd.DataFrame,
        model_density_time_0: u.Quantity,
        model_isotope_time_0: u.Quantity,
        **kwargs,
    ):
        columns = {}
        for column in csv_data.columns:  # Format the dataframe
            if column.lower() in self.valid_fields:
                columns[column] = column.lower()  # Structure
            elif re.match("[a-zA-Z]{1,2}\d*?", column) is not None:  # Element
                columns[column] = column.lower().capitalize()
            else:
                raise ValueError(
                    f"Column {column} does not appear to be a valid token!"
                )

        self.csv_data = csv_data[columns].rename(columns=columns)
        fields = [CSVYField(name=column) for column in columns.values()]
        self.metadata = CSVYHeader(
            model_density_time_0=model_density_time_0,
            model_isotope_time_0=model_isotope_time_0,
            datatype={"fields": fields},
            **kwargs,
        )

    @staticmethod
    def dict_factory(data):
        """Convienience method for stripping away unused attributes"""
        return {
            k: str(v) if isinstance(v, u.Quantity) else v
            for k, v in data
            if v is not None
        }

    def write_header(self, buffer):
        """create the YAML part of the CSVY file
        returns the string to write to the file"""

        yaml.dump(
            asdict(self.metadata, dict_factory=self.dict_factory),
            buffer,
            explicit_start=True,
            sort_keys=False,
        )

    def write_csv(self, buffer, mode="a"):
        """Create the csvy part of the file
        uses a stream for getting the raw data"""

        self.csv_data.to_csv(buffer, mode=mode, index=False)

    @property
    def header(self):
        stream = io.StringIO()
        self.write_header(stream)
        return stream.getvalue()

    @property
    def csv(self):
        stream = io.StringIO()
        self.write_csv(stream, mode="w")
        return stream.getvalue()

    def write(self, file_or_buff, mode="w"):
        if hasattr(file_or_buff, "write"):
            self.write_header(file_or_buff)
            file_or_buff.write("---\n")
            self.write_csv(file_or_buff, mode="a")
        else:
            with open(file_or_buff, mode) as f:
                self.write(f)


if __name__ == "__main__":
    model_isotope_time_0 = 5 * u.day
    model_density_time_0 = 100 * u.s
    v_inner_boundary = 1e9 * u.cm / u.s

    import numpy as np

    data = np.random.random((10, 10))
    header = ["velocity", "density", "O", "C", "Mg", "ca", "ni56", "FE", "Si", "s"]
    df = pd.DataFrame(data, columns=header)

    writer = CSVYWriter(
        df,
        name="my_csvy_model",
        model_density_time_0=model_density_time_0,
        model_isotope_time_0=model_isotope_time_0,
        v_inner_boundary=v_inner_boundary,
    )

    print(writer.header)
    print(writer.csv)

    writer.write("/tmp/csvy_test.csvy")
