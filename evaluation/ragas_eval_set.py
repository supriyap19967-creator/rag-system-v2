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
        "query": "Identify the trend shown in Figure 5.1 regarding the adoption of voluntary standards.",
        "ground_truth": (
            "Voluntary standards adoption shows a steady upward trend in lower-income countries compared to high-income countries."
        ),
        "category": "visual_extraction",
    },
]
