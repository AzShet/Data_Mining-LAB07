
import pytest
import pandas as pd
from main_module import clean_data, calculate_iv_optimized, select_strong_predictors

@pytest.fixture
def sample_df():
    """
    Proporciona un dataframe de ejemplo limpio para pruebas.
    """
    data = {
        'clump_thickness': [1, 2, 3, 4],
        'uniformity_cell_size': [1, 2, 1, 2],
        'bare_nuclei': [1, 2, 3, 4],
        'class': [0, 1, 0, 1],
        'id': [123, 456, 789, 101]
    }
    return pd.DataFrame(data)

def test_clean_data(sample_df):
    """
    Prueba la limpieza de datos, asegurando que la columna 'id' se elimine.
    """
    cleaned = clean_data(sample_df)
    assert 'id' not in cleaned.columns
    assert cleaned['bare_nuclei'].dtype == int
    assert set(cleaned['class'].unique()).issubset({0, 1})

def test_calculate_iv_optimized(sample_df):
    """
    Prueba el cálculo de IV asegurando que se obtengan valores positivos.
    """
    df = clean_data(sample_df)
    ivs = calculate_iv_optimized(df, 'class')
    assert all(iv >= 0 for iv in ivs)

def test_select_strong_predictors(sample_df):
    """
    Prueba la selección de variables con IV mayor al umbral.
    """
    df = clean_data(sample_df)
    ivs = calculate_iv_optimized(df, 'class')
    selected = select_strong_predictors(ivs, threshold=0.0)
    assert len(selected) > 0
