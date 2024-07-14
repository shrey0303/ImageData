# Image Data

---

## Description

---

Aim of this project is to implement a graph label propagation method to study and classify Image data.

<img width="450" alt="figure" src="https://github.com/shrey0303/ImageData/blob/master/figures/accuracies1.png">
---

## Virtual environment

---

Use the following command lines to create and use venv python package:

```
python3.10 -m venv venv
```

Then use the following to activate the environment:

```
source venv/bin/activate
```

You can now use pip to install any packages you need for the project and run python scripts, usually through a `requirements.txt`:

```
python -m pip install -r requirements.txt
```

When you are finished, you can stop the environment by running:

```
deactivate
```

---

## Project Organization

---

    ├── README.md          -- Top-level README.
    │
    ├── notebooks          -- Jupyter notebooks.
    │
    ├── articles           -- Related articles and useful references.
    │
    ├── reports            -- Notes and report (Latex, pdf).
    │ 
    ├── figures            -- Optional graphics and figures to be included in the report.
    │
    ├── data               -- data sets.
    │
    ├── model_saves        -- stored trained models.
    │
    ├── requirements.txt   -- Requirements file for reproducibility.
    │
    └── src                -- Source code for use in this project.
        │
        ├── __init__.py    -- Makes src a Python package (and not just a module)
        │
        ├── random_forest  -- script for the random forest regressor
        │
        ├── st_app         -- structure for streamlit application
        │
        ├── st_utils       -- helper functions called in st_app.py
        │
        ├── helpers        -- various small helper functions
        │
        └── visualization  -- visual features for exploratory data analysis
