from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

person: Person = {"name": 'Sachin', 'age': 30}

print("Type of person variable", type(person))

print(person)

person: Person = {"name": 'Sachin', 'age': '30'}

print("Type of person variable", type(person))

print(person)