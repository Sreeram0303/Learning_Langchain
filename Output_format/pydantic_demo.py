from pydantic import BaseModel, EmailStr, Field
from typing import Optional
# Pydantic is a data validation and settings management library for Python, which uses Python type annotations.
# It allows you to define data models with type annotations and provides automatic validation and serialization.
class Student(BaseModel):
    name: str #pydantic helps to validate the data type of the field
    age: int = 18 # default value for age
    # email: Optional[str] = None # Optional field, can be None
    email: EmailStr
    cgpa : float = Field(gt=0,lt=11)
new_student = {"name" : "John Doe","email" :"123@gmail.com","cgpa" : 5 }
student = Student(**new_student)
print(student)

student_dict = dict(student)
print(student_dict['age'])
student_json = student.model_dump_json()
print(student_json)