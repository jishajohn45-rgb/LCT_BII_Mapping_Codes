# LCT-BII Mapping and Sensitivity Analysis

This repository contains the Python implementation for mapping Global Terrestrial Vertebrate Species Richness and calculating the **Land Cover Transition Biodiversity Integrity Index (LCT-BII)**.

## Data Availability
To comply with data licensing restrictions from **BirdLife International** and **IUCN**, the raw species distribution data are not included in this repository. 

**How to test the code:**
The provided scripts automatically detect if the raw data is missing. If missing, they generate a **synthetic 'mock' dataset** to demonstrate the computational workflow, LCT-BII formulation, and Monte Carlo sensitivity analysis.

## Installation
Ensure you have Python 3.9+ installed. Install dependencies via:
```bash
pip install -r requirements.txt