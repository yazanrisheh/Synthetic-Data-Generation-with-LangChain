from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import time

from ollama_model import EmployeeRecord

load_dotenv()

llm = Ollama(model="llama3", temperature=0.7)
# llm = ChatOllama(model="llama3", temperature=0.7)


examples = [
    {
        "example": """Employee ID: 10234, Employee Name: Alice Johnson, Department: 
        Marketing, Role: Senior Manager, Salary: $85000, Performance Rating: 4"""
    },
    {
        "example": """Employee ID: 20456, Employee Name: Raj Patel, Department: 
        Finance, Role: Analyst, Salary: $65000, Performance Rating: 3"""
    },
    {
        "example": """Employee ID: 30567, Employee Name: Miguel Hernandez, Department: 
        IT, Role: Software Developer, Salary: $95000, Performance Rating: 5"""
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

def create_data_generator(output_schema, llm, prompt, **kwargs):
    """
    Creates a SyntheticDataGenerator for generating structured data.
    """
    return SyntheticDataGenerator(
        template=prompt,
        llm=llm,
        output_schema=output_schema,
        **kwargs,
    )

synthetic_data_generator = create_data_generator(
    output_schema=EmployeeRecord,
    llm=llm,
    prompt=prompt_template
)


def generate_synthetic_data(runs=1):
  """
  Generates synthetic employee records using the Base SyntheticDataGenerator.
  """
  start_time = time.time()
  synthetic_results = synthetic_data_generator.generate(
    subject="employee_record",
    extra="the name must be chosen at random. Make it something you wouldn't normally choose.",
    runs=runs,
  )
  end_time = time.time()

  print(f"Time taken to generate synthetic results: {end_time - start_time:.2f}seconds")
  print(synthetic_results)
  return synthetic_results

results = generate_synthetic_data(runs=1)


