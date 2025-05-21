import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Simulated Gemini LLM async function (stub)
async def gemini_llm(prompt: str) -> str:
    await asyncio.sleep(0.1)  
    return f"LLM response to: {prompt}"

# Tools
class PandasTool:
    @staticmethod
    async def load_csv(file_path: str) -> pd.DataFrame:
        
        await asyncio.sleep(0.1)
        return pd.read_csv(file_path)

#Tool for plotting
class MatplotlibTool:
    @staticmethod
    async def plot_data(df: pd.DataFrame, column: str, output_file: str):
        await asyncio.sleep(0.1) 
        plt.figure()
        df[column].hist()
        plt.title(f"Histogram of {column}")
        plt.savefig(output_file)
        plt.close()

# Agents
class DataFetcher:
    def __init__(self, tool_pandas):
        self.tool_pandas = tool_pandas

    async def fetch(self, csv_path):
        print("[DataFetcher] Fetching data...")
        df = await self.tool_pandas.load_csv(csv_path)
        print("[DataFetcher] Data fetched")
        return df

class Analyst:
    def __init__(self, tool_matplotlib, llm):
        self.tool_matplotlib = tool_matplotlib
        self.llm = llm

    async def analyze(self, df,column):
        
        print(f"[Analyst] Generating visualization for column: {column}")
        output_file = "output_histogram.png"
        await self.tool_matplotlib.plot_data(df, column, output_file)
        print(f"[Analyst] Visualization saved as {output_file}")

        # Use Gemini LLM to summarize data
        prompt = f"Summarize the distribution of the column {column}."
        summary = await self.llm(prompt)
        print(f"[Analyst] LLM summary: {summary}")

        return output_file, summary

# Group Chat Coordination 
class RoundRobinGroupChat:
    def __init__(self, agents):
        self.agents = agents
        self.index = 0

    async def run(self, csv_path):
        
        df = await self.agents[self.index].fetch(csv_path)
        self.index = (self.index + 1) % len(self.agents)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available to analyze.")
        print("Available numeric columns:")
        for col in numeric_cols:
            print(f" - {col}")

        selected_col = None
        while selected_col not in numeric_cols:
            selected_col = input("Enter the column name to visualize: ").strip()
            if selected_col not in numeric_cols:
                print("Invalid column name. Please select from the list above.")

        output_file, summary = await self.agents[self.index].analyze(df, selected_col)
        self.index = (self.index + 1) % len(self.agents)
        return output_file, summary


# Main async function
async def main():
    pandas_tool = PandasTool()
    matplotlib_tool = MatplotlibTool()

    data_fetcher = DataFetcher(pandas_tool)
    analyst = Analyst(matplotlib_tool, gemini_llm)

    agents = [data_fetcher, analyst]
    group_chat = RoundRobinGroupChat(agents)

    csv_path = "cs.csv"  

    try:
        output_file, summary = await group_chat.run(csv_path)
        print(f"\nPipeline complete! Output: {output_file}")
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"Error in pipeline: {e}")

if __name__ == "__main__":
    asyncio.run(main())
