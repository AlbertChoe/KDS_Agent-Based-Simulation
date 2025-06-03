# KDS Agent-Based Simulation

This project implements an agent-based simulation system for modeling complex systems. The simulation focuses on ecological and geographical interactions using real-world datasets.

## Features
- Agent-based modeling framework
- Geographical data integration (Galapagos GADM)
- Species interaction modeling
- Interactive visualization
- Parameter configuration system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlbertChoe/KDS_Agent-Based-Simulation.git
cd KDS_Agent-Based-Simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation:
```bash
streamlit run ui.py
```

The application provides:
- Interactive visualization
- Configuration options
- Data export capabilities

## Data Sources
- Geographical data: `gadm41_ECU_2.json`
- Species data: `species_data.json`
- Additional documentation: `data.md`

## Project Structure
```
.
├── README.md               - This file
├── main.py                 - Main application entry point
├── ui.py                   - User interface components
├── requirements.txt        - Python dependencies
├── .gitignore              - Git ignore rules
├── data.md                 - Data documentation
├── gadm41_ECU_2.json       - Geographical dataset
└── species_data.json       - Species dataset
```

## Contributor
- Juan Alfred Widjaya / 13522073
- Albert / 13522081
- Ivan Hendrawan Tan / 13522111