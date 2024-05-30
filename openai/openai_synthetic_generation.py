from save_openai_to_csv import save_data_to_csv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import (
    create_openai_data_generator
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

from openai_model import MedicalBilling

load_dotenv()

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.7)


examples = [
    {
        "example": """Patient ID: 123456, Patient Name: John Doe, Diagnosis Code:
        J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"""
    },
    {
        "example": """Patient ID: 789012, Patient Name: Johnson Smith, Diagnosis
        Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"""
    },
    {
        "example": """Patient ID: 345678, Patient Name: Emily Stone, Diagnosis Code:
        E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"""
    },
]

OPENAI_TEMPLATE = PromptTemplate.from_template(template="{example}")
prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

synthetic_data_generator = create_openai_data_generator(
    output_schema=MedicalBilling,
    llm=llm,
    prompt=prompt_template,
)

def generate_synthetic_data(runs=1):
    """Generates synthetic data."""

    start_time = time.time()
    synthetic_results = synthetic_data_generator.generate(
        subject="medical_billing",
        extra="The name must be chosen at random. Make it something you wouldn't normally choose.",
        runs=runs,
    )
    end_time = time.time()
    print(f"Time taken to generate synthetic results: {end_time - start_time:.2f} seconds")
    return synthetic_results


results = generate_synthetic_data(runs=50)
save_data_to_csv(results, filename="synthetic_data.csv")