from candidate import Candidate

def test_from_dict() -> None:
    data = {
        "name": "Alice",
        "skills": ["React", "TypeScript"],
        "experience": 2
    }

    candidate = Candidate.from_dict(data)

    assert candidate.name == "Alice"
    assert candidate.skills == ["React", "TypeScript"]
    assert candidate.experience == 2

def test_to_dict() -> None:
    candidate = Candidate(
        name="Bob",
        skills=["Python"],
        experience=3,
    )

    result = candidate.to_dict()

    assert result == {
        "name": "Bob",
        "skills": ["Python"],
        "experience": 3,
    }


def test_empty_skills() -> None:
    candidate = Candidate(
        name="Charlie",
        skills=[],
        experience=0,
    )

    assert candidate.skills == []

def test_print_profile(capsys) -> None:
    candidate = Candidate(
        name="Alice",
        skills=["React", "TypeScript"],
        experience=2,
    )

    candidate.print_profile()

    captured = capsys.readouterr()

    assert captured.out == (
        "候选人：Alice\n"
        "经验：2 年\n"
        "\n"
        "技能：\n"
        "- React\n"
        "- TypeScript\n"
    )