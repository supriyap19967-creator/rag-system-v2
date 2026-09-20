# evaluation/step_1_dataset.py
# Multimodal Evaluation Dataset for Offline Benchmarking
# Populated with 7 concrete sample items representing every single and hybrid modality combination

multimodal_eval_dataset = [
    {
        "id": "eval-001",
        "question": "Why are standards important for developing countries?",
        "modality": ["text"],
        "inputs": {
            "image_path": None,
            "csv_path": None,
            "text_context": "Standards are essential for developing countries because they lower transaction costs, signal quality to foreign buyers, and facilitate the transfer of knowledge regarding international best practices."
        },
        "expected_output": "Standards lower transaction costs, signal product quality to international buyers, and facilitate the transfer of best-practice knowledge."
    },
    {
        "id": "eval-002",
        "question": "What is the average GDP for India in the dataset?",
        "modality": ["csv"],
        "inputs": {
            "image_path": None,
            "csv_path": "Data/world_development_report_2025_gdp_co2.csv",
            "text_context": None
        },
        "expected_output": 2747952252627.51
    },
    {
        "id": "eval-003",
        "question": "Identify the trends shown in Figure 5.1 regarding vaccine coverage and child mortality.",
        "modality": ["visual"],
        "inputs": {
            "image_path": "assets/extracted_images/page_272_Figure_5.1.png",
            "csv_path": None,
            "text_context": None
        },
        "expected_output": "Figure 5.1 shows that vaccine coverage has increased significantly while the under-five mortality rate has steadily declined between 1974 and 2023."
    },
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
            "image_path": "assets/extracted_images/page_302_Figure_6.1.png",
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
            "image_path": "assets/extracted_images/page_207_Table_4.1.png",
            "csv_path": "Data/world_development_report_2025_gdp_co2.csv",
            "text_context": None
        },
        "expected_output": "The CSV emissions data indicates rising trends for India, while Table 4.1 outlines firm-level impacts of standards adoption, showing positive outcomes across studies for sales growth, profit growth, and exports."
    },
    {
        "id": "eval-007",
        "question": "Synthesize the environmental impacts of localized air pollution using the CSV metrics, the textual policy recommendations, and the coexistence network of policy instruments in Figure 6.1.",
        "modality": ["text", "csv", "visual"],
        "inputs": {
            "image_path": "assets/extracted_images/page_302_Figure_6.1.png",
            "csv_path": "Data/world_development_report_2025_gdp_co2.csv",
            "text_context": "Place-based standards for localized air pollution often lead to emissions leakage where strict rules in one region shift production to less regulated ones."
        },
        "expected_output": "Place-based localized air pollution rules can trigger emissions leakage, which can be addressed by coexisting policy instruments such as those illustrated in Figure 6.1's network."
    }
]

if __name__ == "__main__":
    print(f"Loaded multimodal_eval_dataset with {len(multimodal_eval_dataset)} test cases.")
    for case in multimodal_eval_dataset:
        print(f"ID: {case['id']} | Modality: {case['modality']} | Question: {case['question']}")
