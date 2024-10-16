from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# loading the api keys from .env
load_dotenv() 
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
result = model.invoke("hello!")
print(result.content)
# print(result.content)

