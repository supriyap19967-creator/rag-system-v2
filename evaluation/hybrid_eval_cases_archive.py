"""
Preserved Archive of Hybrid Cross-Modal RAGAS Evaluation Cases.
Saved separately for future reference if hybrid/cross-modal evaluation is revisited.
"""

HYBRID_EVALUATION_CASES = [
    {
        "query": "What was India GDP in 2022 and what does the report say about economic growth?",
        "ground_truth": (
            "The answer should include India's validated 2022 GDP value and a grounded explanation from the PDF "
            "about economic growth conditions."
        ),
        "category": "hybrid_mixed",
    },
    {
        "query": "What was India GDP and CO2 emission in 2022 and explain their impact?",
        "ground_truth": (
            "The answer should include validated 2022 GDP and CO2 values for India plus a grounded explanation "
            "linking growth and environmental pressure."
        ),
        "category": "hybrid_multi_metric",
    },
]

HYBRID_MULTIMODAL_DATASET_CASES = [
    {
        "id": "eval-004",
        "question": "Calculate the difference in standards adoption rate and explain why it varies based on country income level.",
        "modality": ["text", "csv"],
        "inputs": {
            "image_path": None,
            "csv_path": "Data/world_development_report_2025_gdp_co2.csv",
            "text_context": "The adoption rate of standards varies because lower-income countries often face higher compliance costs and lower capacity to implement voluntary standards than high-income countries."
        },
        "expected_output": "The difference is 12% and the variation is due to compliance costs and capacity limitations in lower-income countries."
    },
    {
        "id": "eval-005",
        "question": "Explain standard harmonization using the details in the text and the coexistence network depicted in Figure 6.1.",
        "modality": ["text", "visual"],
        "inputs": {
            "image_path": "extracted_images/page_302_Figure_6.1.png",
            "csv_path": None,
            "text_context": "Harmonization of standards reduces trade barriers by ensuring mutual recognition of test results across regional boundaries."
        },
        "expected_output": "Standard harmonization reduces trade barriers, complementing the coexistence of environmental standards and other policy instruments shown in Figure 6.1."
    },
    {
        "id": "eval-006",
        "question": "Compare the numeric emissions data from the CSV with the standards adoption impacts represented in Table 4.1.",
        "modality": ["csv", "visual"],
        "inputs": {
            "image_path": "extracted_images/page_208_Table_4.1.csv",
            "csv_path": "Data/world_development_report_2025_gdp_co2.csv",
            "text_context": None
        },
        "expected_output": "The CSV emissions data indicates rising trends for India, while Table 4.1 outlines firm-level impacts of standards adoption, including positive effects on exports and sales."
    },
    {
        "id": "eval-007",
        "question": "Synthesize the environmental impacts of localized air pollution using the CSV metrics, the textual policy recommendations, and the coexistence network of policy instruments in Figure 6.1.",
        "modality": ["text", "csv", "visual"],
        "inputs": {
            "image_path": "extracted_images/page_302_Figure_6.1.png",
            "csv_path": "Data/world_development_report_2025_gdp_co2.csv",
            "text_context": "Place-based standards for localized air pollution often lead to emissions leakage where strict rules in one region shift production to less regulated ones."
        },
        "expected_output": "Place-based localized air pollution rules can trigger emissions leakage, which can be addressed by coexisting policy instruments such as those illustrated in Figure 6.1's network."
    }
]
