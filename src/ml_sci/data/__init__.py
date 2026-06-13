"""Synthetic scientific DGP exports."""

from ml_sci.data.base import ClimateDataset, GenomicsDataset, MaterialsDataset, ProteinDataset
from ml_sci.data.climate_dgp import ClimateDGPConfig, generate_climate_data
from ml_sci.data.genomics_dgp import GenomicsDGPConfig, generate_genomics_data
from ml_sci.data.materials_dgp import MaterialsDGPConfig, generate_materials_data, property_oracle
from ml_sci.data.protein_dgp import ProteinDGPConfig, generate_protein_data, stability_oracle

__all__ = [
    "ClimateDGPConfig",
    "ClimateDataset",
    "GenomicsDGPConfig",
    "GenomicsDataset",
    "MaterialsDGPConfig",
    "MaterialsDataset",
    "ProteinDGPConfig",
    "ProteinDataset",
    "generate_climate_data",
    "generate_genomics_data",
    "generate_materials_data",
    "generate_protein_data",
    "property_oracle",
    "stability_oracle",
]
