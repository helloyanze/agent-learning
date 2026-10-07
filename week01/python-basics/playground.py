skills = ["React", "TypeScript", "Python"]

# %% A：skills 内容转大写
skills_upper = [skill.upper() for skill in skills]
print(skills_upper)  # 输出: ['REACT', 'TYPESCRIPT', 'PYTHON']


# %% B:计算所有参数的和
def add_all(*args: int) -> int:
    print(type(args))
    return sum(args)


print(add_all(1, 2, 3))
print(add_all(10, 20, 30, 40))


# %% C:**kwargs
def create_user(**kwargs) -> dict:
    print(type(kwargs))
    return kwargs


user = create_user(
    name="Alice",
    age=20,
    city="Wuhan",
)

print(user)

# %% D:try/except
# number = int("abc")
try:
    number = int("abc")
except ValueError as e:
    print(f"转换失败: {e}")

# %% E:with
with open("hello.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!")

with open("hello.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)  # 输出: Hello, World!

# %% F: async / await
import asyncio


async def fetch_user() -> str:
    # 等待 1 秒
    # 返回 "Alice"
    await asyncio.sleep(1)
    return "Alice"


async def main() -> None:
    user = await fetch_user()
    print(user)


asyncio.run(main())
