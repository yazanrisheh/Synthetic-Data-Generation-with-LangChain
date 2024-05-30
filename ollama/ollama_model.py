from langchain_core.pydantic_v1 import BaseModel

class EmployeeRecord(BaseModel):
    employee_id: int
    employee_name: str
    department: str
    role: str
    salary: float
    performance_rating: int