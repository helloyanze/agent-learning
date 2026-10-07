from typing import Any

class Candidate:

    def __init__(
        self,
        name: str,
        skills: list[str],
        experience: int
    ) -> None:
        self.name = name
        self.skills = skills
        self.experience = experience

    @classmethod
    def from_dict(cls, data: dict[str,Any]) -> "Candidate":
        return cls(
            name=data["name"],
            skills=data["skills"],
            experience=data["experience"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "skills": self.skills,
            "experience": self.experience,
        }

    def print_profile(self) -> None:
        # 候选人：Alice
        # 经验：2 年

        # 技能：
        # - React
        # - TypeScript
        print(
            f"候选人: {self.name}\n"
            f"经验: {self.experience}年\n"
            "\n"
            "技能：\n"
            + "\n".join(f"- {skill}" for skill in self.skills)
        )
            
