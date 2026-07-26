EVALUATION_CASES = [
    {
        "query": "What was India GDP in 2022?",
        "ground_truth": "In 2022, GDP (current US$) for India (IND) was 3346107287730.93.",
        "category": "csv_factual",
    },
    {
        "query": "Why are standards important for developing countries?",
        "ground_truth": (
            "Standards help developing countries by spreading good practices, improving quality and efficiency, "
            "and supporting trade, investment, growth, and risk management."
        ),
        "category": "pdf_explanatory",
    },
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
