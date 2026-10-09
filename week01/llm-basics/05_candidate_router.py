import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


class CandidateModel(BaseModel):
    name: str
    skills: list[str]
    experience: int


# 如果 skills 中存在：
# React / TypeScript / Vue
# → 返回 "frontend"
# 如果存在：
# Python / FastAPI / Django
# → 返回 "backend"
# 否则：
# → 返回 "unknown"
def classify_candidate(candidate: CandidateModel) -> str:
    skills = {skill.lower() for skill in candidate.skills}

    frontend_skills = {"react", "typescript", "vue"}
    backend_skills = {"python", "fastapi", "django"}

    has_frontend = bool(skills & frontend_skills)
    has_backend = bool(skills & backend_skills)

    if has_frontend and has_backend:
        return "fullstack"

    if has_frontend:
        return "frontend"

    if has_backend:
        return "backend"

    return "unknown"
