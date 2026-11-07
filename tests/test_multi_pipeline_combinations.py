import pytest
import pandas as pd
from app.intent_router import classify_query_intent, QueryIntent
from app.agents.synthesizer import _clean_scientific_notation_in_text
from streamlit_ui.StreamlitApp import clean_dataframe_for_table_rendering


def test_modality_1_data_and_text_classification():
    query = "Compare the average GDP of low-income countries in the table with the three types of standards explained in the report"
    intent, tools = classify_query_intent(query)
    assert intent == QueryIntent.HYBRID_MULTIMODAL
    assert tools["allow_pandas"] is True
    assert tools["allow_qdrant"] is True
    assert tools["allow_vision"] is False, "Vision must be disabled for pure Data + Text queries"


def test_modality_2_text_and_visual_classification():
    query = "Compare the definition of standards in Chapter 4 with the adoption metrics in Figure 4.1"
    intent, tools = classify_query_intent(query)
    assert intent == QueryIntent.HYBRID_MULTIMODAL
    assert tools["allow_pandas"] is False, "Pandas must be disabled for pure Text + Visual queries"
    assert tools["allow_qdrant"] is True
    assert tools["allow_vision"] is True


def test_modality_3_data_and_visual_classification():
    query = "Compare top 5 countries by CO2 in 2020 with the adoption trends in Figure 4.1"
    intent, tools = classify_query_intent(query)
    assert intent == QueryIntent.HYBRID_MULTIMODAL
    assert tools["allow_pandas"] is True
    assert tools["allow_vision"] is True


def test_modality_4_tri_modal_classification():
    query = "Compare top 5 GDP nations in 2023 with mortality rates in Figure 5.1 and health policy arguments"
    intent, tools = classify_query_intent(query)
    assert intent == QueryIntent.HYBRID_MULTIMODAL
    assert tools["allow_pandas"] is True
    assert tools["allow_qdrant"] is True
    assert tools["allow_vision"] is True


def test_scientific_notation_cleanup():
    raw_text = "The United States GDP was 2.72922e+13 in 2023 and Germany was 4.56221e+12."
    cleaned = _clean_scientific_notation_in_text(raw_text)
    assert "2.72922e+13" not in cleaned
    assert "$27.29 Trillion" in cleaned
    assert "$4.56 Trillion" in cleaned


def test_clean_dataframe_drops_dummy_placeholder_rows():
    df = pd.DataFrame([
        {"Series": "100", "Category": "100", "Value": "A", "Target Value": "100"},
        {"Series": "United States", "Category": "High Income GDP", "Value": "$27.29 Trillion", "Target Value": "27.29T"},
    ])
    cleaned_df = clean_dataframe_for_table_rendering(df)
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["Series"] == "United States"
