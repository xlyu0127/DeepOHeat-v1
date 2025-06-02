# DeepOHeat-v1

## 📁 Project Structure
- `heat_surface.py` – Solves the heat equation parameterized by surface power distributions.
- `heat_volumetric.py` – Solves the heat equation parameterized by volumetric power distributions.
- `hybrid_solver.ipynb` – Hybrid solver combining operator learning model and tnaditional umerical method (GMRES).
- `train.py`, `models.py` – Core model definition and training scripts.
- `eval.py` – Evaluation utilities.

## 🚀 Getting Started

### Download the Data
[Download from Google Drive](https://drive.google.com/drive/folders/13g2dkNU1AU0OPGRPvkBAncguAK7Cb6Ek?usp=sharing)

### Run Surface Power Heat Simulation
```bash
python heat_surface.py
```

### Run Volumetric Power Heat Simulation
```bash
python heat_volumetric.py
```