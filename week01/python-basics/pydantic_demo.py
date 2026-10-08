from pydantic import BaseModel


class CandidateModel(BaseModel):
    name: str
    skills: list[str]
    experience: int


# 1. 创建一个候选人对象
# candidate = CandidateModel(
#     name="Alice",
#     skills=["React", "Python"],
#     experience=2,
# )

# print(candidate)
# print(candidate.name)
# print(candidate.skills)
# print(candidate.experience)

# 2. 创建一个候选人对象，experience 为字符串类型
# candidate = CandidateModel(
#     name="Alice",
#     skills=["React"],
#     experience="2",
# )

# print(candidate)
# print(type(candidate.experience))

# 3. 创建一个候选人对象，experience 为非数字字符串类型
# candidate = CandidateModel(
#     name="Alice",
#     skills=["React"],
#     experience="abc",
# )

# print(candidate)

# 4. 创建一个候选人对象，skills 为字符串类型
# candidate = CandidateModel(
#     name="Alice",
#     skills="Python",
#     experience=2,
# )

# candidate = CandidateModel(
#     name="Alice",
#     skills=[1, 2, 3],
#     experience=2,
# )

# 5. Pydantic 对象 -> dict
# candidate = CandidateModel(
#     name="Alice",
#     skills=["React", "Python"],
#     experience=2,
# )

# data = candidate.model_dump()

# print(data)
# print(type(data))


# 6. Pydantic 对象 -> JSON 字符串
# json_data = candidate.model_dump_json()

# print(json_data)
# print(type(json_data))


# 7. dict -> Pydantic 对象
data = {
    "name": "Bob",
    "skills": ["Python", "FastAPI"],
    "experience": 3,
}

candidate = CandidateModel.model_validate(data)

print(candidate)
print(type(candidate))
