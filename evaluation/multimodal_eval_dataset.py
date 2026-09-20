# Multimodal Evaluation Dataset for Single-Modality Benchmark Evaluation
# Populated with concrete sample items representing individual supported query paths

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
        "question": "Identify the trend shown in Figure 5.1 regarding the adoption of voluntary standards.",
        "modality": ["visual"],
        "inputs": {
            "image_path": "extracted_images/page_272_Figure_5.1.png",
            "csv_path": None,
            "text_context": None
        },
        "expected_output": "Voluntary standards adoption shows a steady upward trend in lower-income countries compared to high-income countries."
    }
]

if __name__ == "__main__":
    print(f"Loaded multimodal_eval_dataset with {len(multimodal_eval_dataset)} test cases.")
    for case in multimodal_eval_dataset:
        print(f"ID: {case['id']} | Modality: {case['modality']} | Question: {case['question']}")
