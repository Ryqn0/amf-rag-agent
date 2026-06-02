from amf_rag_agent import config
from langsmith import Client


client = Client()

dataset = client.create_dataset(
    dataset_name="amf-rag-eval",
    description="AMF filing questions with reference answers",  
)


examples = [
    {
        "inputs": {"question": "What does BNP Paribas say about TCFD recommendations"},
        "outputs": {"answer": "BNP Paribas includes a TCFD concordance table in its sustainability reporting, aligns its governance and climate risk management with TCFD principles, and reports alongside GRI and SASB standards."}
    },
    {
        "inputs": {"question": "What are the main risk factors for TotalEnergies?"},
        "outputs": {"answer": "The main risk factors for TotalEnergies include energy transitions and strategic risks, market and financial risks, geopolitical and security risks, climate and operational risks, cibersecurity risks, and Health, Safety and Environment (HSE) risks."}
    },
    {
        "inputs": {"question": "What was LVMH's revenue in 2024?"},
        "outputs": {"answer": "LVMH's revenue in 2024 was €84,683 million (approximately €84.7 billion)."}
    },
    {
        "inputs": {"question": "How much did BNP Paribas deploy toward low-carbon transition?"},
        "outputs": {"answer": "BNP Paribas deployed €215 billion toward the low-carbon transition in 2024."}
    },
    {
        "inputs": {"question": "How do TotalEnergies and LVMH each use TCFD recommendations?"},
        "outputs": {"answer": "TotalEnergies uses TFCD as a framework for mapping and positioning risks at the enterprise risk management level focusing on physical and transition risks of fossil fuel operations, while LVMH uses TCFD as a part of its LIFE 360 sustainability program integrating environmental considerations and double materiality."}
    },
    {
        "inputs": {"question": "Que dit BNP Paribas sur la gestion des risques climatiques?"},
        "outputs": {"answer": "BNP Paribas considère les risques climatiques comme des facteurs potentiels des catégories de risque financiers. Les piliers de gestion des risques climatiques comprennent le renforcement du cadre de stress-test climatique, la gestion des risques physiques et de transition, la politique de financement durable, l'engagement de soi et auprès des clients, et la divulgation."}
    },
    {
        "inputs": {"question": "What sustainability reporting standards do BNP Paribas, Airbus, LVMH and TotalEnergies companies follow?"},
        "outputs": {"answer": "For BNP PARIBAS, the sustainability reporting standards followed include ESRS (European Sustainability Reporting Standards), SASB (Sustainability Accounting Standards Board), TCFD (Task Force on Climate-related Financial Disclosures), EU Taxonomy and CSRD (Corporate Sustainability Reporting Directive). LVMH follows ESRS, CSRD, and DMA (Double Materiality Assessment) with ESRS expectations. TotalEnergies follows ESRS, GRI, SASB, TCFD, World Economic Forum, CDP water, CSRD and EU Taxonomy. Airbus follows ESRS."}
    },
    {
        "inputs": {"question": "What does TotalEnergies report about its ESG databook?"},
        "outputs": {"answer": "TotalEnergies's ESG databook provides comprehensive performance indicatos, reporting standards according to GRI (Global Reporting Initiative), SASB (Sustainability Accounting Standards Board) and World Economic Forum, climate and environmental frameworks relying on TCFD (Task Force on Climate-related Financial Disclosures) and CDP water questionnaires, accessibility and Stakeholder Engagement."}
    },
    {
        "inputs": {"question": "What are the main risk factors Airbus identifies?"},
        "outputs": {"answer": "Airbus identifies several main risk factors, including energy transition risks, market environment parameters risks, geopolitical and external risks, operational risks concerning supply chain, production and technical qualifications risks, technological and innovation risks."}
    }
]


client.create_examples(dataset_id=dataset.id, examples=examples)
print(f"Created dataset with ID: {dataset.id}")
print(f"Example questions with {len(examples)} examples added.")