from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Person(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr = "nouser@email.com"
    enable2FA: bool = Field( default=False)

person = Person(**{"name": 'Sachin', 'age': 30})
print("Type of person variable", type(person))
print(person)

person = Person(**{"name": 'Golu', 'age': '30'})
print("Type of person variable", type(person))
print(person)

person = Person(**{"name": 'Jitu'})
print("Type of person variable", type(person))
print(person)

# Error code
person = Person(**{"name": 78, 'age': '30'})
print("Type of person variable", type(person))
print(person)