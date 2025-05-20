import os
import asyncio
import tempfile
import subprocess
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load Gemini API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

genai.configure(api_key=GEMINI_API_KEY)


class Coder:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def run(self, context):
        print("\n[Coder is generating code...]\n")
        prompt = f"Write Python code for the following task:\n{context}"
        response = await asyncio.to_thread(self.model.generate_content, prompt)
        full_text = response.text

        code_blocks = re.findall(r"```python(.*?)```", full_text, re.DOTALL)
        
        if code_blocks:
            # Limit to 2 code blocks
            limited_code = "\n\n".join(code_blocks[:2]).strip()
        else:
        
            parts = re.split(r"(?=def\s)", full_text)
            
            # Clean empty entries
            parts = [p.strip() for p in parts if p.strip()]
            
            # Take first two function blocks and maybe some following usage code 
            if len(parts) >= 2:
                limited_code = "\n\n".join(parts[:2])
            else:
                limited_code = full_text[:1000].strip()

        print("[Generated Code]\n", limited_code)
        return limited_code



# Debugger agent
class Debugger:
    async def run(self, code):
        print("\n[Debugger is analyzing the code...]\n")
        code = re.sub(r'input\([^)]+\)', '"7"', code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(code)
            tmpfile_path = tmpfile.name

        try:
            pylint_output = subprocess.check_output(
                ["pylint", tmpfile_path, "--disable=all", "--enable=E,F"],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        except subprocess.CalledProcessError as e:
            pylint_output = e.output

        try:
            execution_output = subprocess.check_output(
                ["python", tmpfile_path],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        except subprocess.CalledProcessError as e:
            execution_output = e.output

        result = f"Linter Output \n{pylint_output}\nExecution Output\n{execution_output}"
        print(result)
        return result


# Coordinator class
class RoundRobinGroupChat:
    def __init__(self, agents):
        self.agents = agents

    async def run(self, context):
        data = context
        for agent in self.agents:
            data = await agent.run(data)
        return data


# Main async function
async def main():
    print("=== Gemini Code Generator + Debugger ===")

    coder = Coder()
    debugger = Debugger()
    group_chat = RoundRobinGroupChat([coder, debugger])

    # Step 1: Generate code from task and debug it
    task = input("\n1️. Describe what kind of code you want:\n> ")
    result1 = await group_chat.run(task)

    # Step 2: Ask user to paste their own code
    print("\n2️. Now paste your own Python code (end with empty line):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    user_code = "\n".join(lines)

    result2 = await debugger.run(user_code)

    # Save to file
    with open("debugged_output.txt", "w", encoding="utf-8") as f:
        f.write("=== Generated Code Debugging ===\n")
        f.write(result1 if isinstance(result1, str) else str(result1))
        f.write("\n\n=== User Code Debugging ===\n")
        f.write(result2 if isinstance(result2, str) else str(result2))

    print("\n✅ All done! Results saved to 'debugged_output.txt'")


if __name__ == "__main__":
    asyncio.run(main())
