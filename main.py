from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel

# === CONFIGURATION ===
GEMINI_API_KEY = "AIzaSyBQRb3d-idLAx5L49mbDJoJ-lSjOYBdKU4"
FAISS_FOLDER = "E:/Hackathon/HackIndia/deploy"  # Path to your FAISS index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model for FAISS

# FastAPI setup
app = FastAPI()

# Serve static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup for templates
templates = Jinja2Templates(directory="templates")

# === Initialize Gemini API ===
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    print("✅ Gemini API configured successfully!")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    gemini_model = None

# === Load FAISS DB ===
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
try:
    db = FAISS.load_local(FAISS_FOLDER, embedding, allow_dangerous_deserialization=True)
    print("✅ FAISS DB loaded successfully!")
except Exception as e:
    print(f"❌ Error loading FAISS DB: {e}")
    db = None

# Define the Query model for the POST request
class Query(BaseModel):
    question: str

# === FastAPI Routes ===

# Home route to render the form
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ask route to process the question and return an answer
@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):  # Use Form to extract data from form submission
    if not db:
        return templates.TemplateResponse("index.html", {"request": request, "error": "FAISS database could not be loaded."})
    if not gemini_model:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Gemini model could not be initialized."})

    try:
        # Perform similarity search
        docs = db.similarity_search(question, k=5)  # Search for top 5 similar docs
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create a prompt to pass to Gemini
        prompt = f"""Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"""

        # Get the response from Gemini
        response = gemini_model.generate_content(prompt)

        # Return the answer and sources, render in the HTML template
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "answer": response.text, 
            "sources": [doc.metadata for doc in docs]
        })
    except Exception as e:
        # Handle any errors and return an error message in the template
        return templates.TemplateResponse("index.html", {"request": request, "error": f"An error occurred: {str(e)}"})

# Run FastAPI with `uvicorn app:app --reload`
