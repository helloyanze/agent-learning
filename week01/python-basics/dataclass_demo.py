from dataclasses import dataclass


@dataclass
class CandidateData:
    name: str
    skills: list[str]
    experience: int


candidate = CandidateData(
    name="Alice",
    skills=["React", "Python"],
    experience=2,
)

print(candidate)
print(candidate.name)
print(candidate.skills)
print(candidate.experience)
