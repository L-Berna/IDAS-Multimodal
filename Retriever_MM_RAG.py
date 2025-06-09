"""
Multimodal Search Application for Vehicle Information

This application provides a GUI interface for querying vehicle information using both text and image data.
It combines embeddings from BridgeTower for multimodal representation with OpenAI's GPT for context integration.

Main features:
- Text-based querying with multimodal vector representation
- Fast similarity search using FAISS
- AI-powered re-ranking of search results
- Integration of manual context from ChromaDB
- GUI interface for easy interaction
"""

import os
import io
import re
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import lancedb
import faiss
import threading
from functools import lru_cache
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configure OpenAI API key
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set the 'OPENAI_API_KEY' environment variable with your OpenAI key.")

class VehicleAssistantApp:
    """
    Main application class that handles the multimodal search functionality
    and user interface for the vehicle assistant.
    """
    
    def __init__(self):
        # Connect to LanceDB
        self.lancedb_path = r'C:\Users\krato\Documents\Documentos Uni\3° Semestre\Proyecto\LanceDB_New'
        self.db = lancedb.connect(self.lancedb_path)
        self.table = self.db.open_table("embeddings")
        
        # Load data and build ANN index
        self._build_ann_index()
        
        # Initialize ChromaDB for RetrievalQA
        self._initialize_retrieval_qa()
        
        # Initialize BridgeTower model
        self._initialize_bridgetower()
    
    def _build_ann_index(self):
        """Build Approximate Nearest Neighbor index using FAISS"""
        data = self.table.to_pandas()
        
        if 'vector' not in data.columns:
            raise ValueError("Table does not contain 'vector' column with embedding vectors.")
        
        embeddings = np.array(data['vector'].tolist(), dtype='float32')
        d = embeddings.shape[1]
        
        # Usar IndexHNSWFlat con parámetros optimizados
        self.index_ann = faiss.IndexHNSWFlat(d, 64)  # Aumentar M a 64 para mejor precisión
        self.index_ann.hnsw.efConstruction = 200  # Aumentar para mejor construcción
        self.index_ann.hnsw.efSearch = 100  # Aumentar para mejor búsqueda
        
        # Normalizar vectores antes de añadirlos
        faiss.normalize_L2(embeddings)  # ACTIVADO: Normalización solo en FAISS
        self.index_ann.add(embeddings)
        print(f"FAISS index built with {self.index_ann.ntotal} vectors.")
        
        self.rows_data = data.to_dict('records')
    
    def _initialize_retrieval_qa(self):
        """Initialize ChromaDB and RetrievalQA chain"""
        vectordb_directory = 'vector_database_chspark_1536'
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
        vectordb = Chroma(embedding_function=embedding_model, persist_directory=vectordb_directory)
        
        # Initialize LLM
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4o-mini")
        
        # Configure RetrievalQA with optimized prompt
        template = (
            "You are IDAS, a vehicle expert assistant. Follow these rules:\n"
            "1. MAX 3 bullet points\n"
            "2. MAX 15 words per point\n"
            "3. Technical terms only\n"
            "4. No explanations\n"
            "5. Actionable steps\n\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer:\n"
        )
        qa_chain_prompt = PromptTemplate.from_template(template)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt}
        )
    
    def _initialize_bridgetower(self):
        """
        Initialize BridgeTower model for multimodal embeddings.
        Models are loaded with cache to reduce latency in future initializations.
        """
        self.processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        self.model_bridgetower = BridgeTowerForContrastiveLearning.from_pretrained(
            "BridgeTower/bridgetower-large-itm-mlm-itc"
        ).to(device)
        self.model_bridgetower.eval()
    
    @lru_cache(maxsize=1000)
    def generate_query_embedding(self, query_text):
        """
        Generate a multimodal embedding from text query.
        Uses a dummy image and weights the text embedding (75%) and image embedding (30%),
        without normalizing individual embeddings (normalization only in FAISS).
        
        Args:
            query_text (str): The text query to generate embedding for
            
        Returns:
            list: The generated embedding vector
        """
        dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        encoding = self.processor(
            images=dummy_image,
            text=query_text,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model_bridgetower(**encoding)
            # image_embeds = torch.nn.functional.normalize(outputs.image_embeds, p=2, dim=1)  # COMENTADO: Normalización deshabilitada
            # text_embeds = torch.nn.functional.normalize(outputs.text_embeds, p=2, dim=1)  # COMENTADO: Normalización deshabilitada
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Combine embeddings with optimized weights
            combined_embeds = 0.75 * text_embeds + 0.3 * image_embeds
            # combined_embeds = torch.nn.functional.normalize(combined_embeds, p=2, dim=1)  # COMENTADO: Normalización deshabilitada
        
        return combined_embeds.cpu().numpy().flatten().astype('float32').tolist()
    
    def re_rank_candidates(self, query, candidates):
        """
        Re-rank candidates using the LLM to find the most relevant result.
        
        Args:
            query (str): The original query
            candidates (list): List of candidate results
            
        Returns:
            int: Index of the best candidate
        """
        prompt = f"""Given the following query:
"{query}"

And considering these results:
"""
        for i, cand in enumerate(candidates, start=1):
            prompt += f"{i}. {cand['caption']}\n"
        prompt += "\nChoose the number of the result that best answers the query. Respond with just the number."
        
        response = self.llm(prompt)
        response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        match = re.search(r'\d+', response_text)
        if match:
            try:
                best_number = int(match.group())
                if 1 <= best_number <= len(candidates):
                    return best_number - 1
            except Exception as e:
                print("Error converting response to integer:", e)
        
        print("Error in re-ranking, unexpected response:", response_text)
        return 0
    
    def is_vehicle_related(self, query_text):
        """
        Determina si una consulta está relacionada con vehículos o manuales técnicos.
        
        Args:
            query_text (str): Consulta del usuario.
        
        Returns:
            bool: True si la consulta es sobre vehículos, de lo contrario False.
        """
        prompt = f"""
Determine si la siguiente consulta está relacionada con vehículos, manuales de vehículos, 
o información técnica sobre automóviles.

Consulta: "{query_text}"

Responde únicamente con "SI" si está relacionada con vehículos o "NO" si no lo está.
"""
        response = self.llm(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response).strip()
        return "SI" in response_text.upper()
    
    def search_by_text(self, query_text, top_k=5):
        """
        Perform a search by generating a multimodal embedding and using FAISS to get top-K records,
        then apply re-ranking to select the most relevant one, and finally integrate manual context and image.
        
        Args:
            query_text (str): The text query
            top_k (int): Number of top results to retrieve
            
        Returns:
            tuple: (best_result, response_text)
        """
        # Generate embedding and search
        query_embedding = self.generate_query_embedding(query_text)
        query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
        distances, indices = self.index_ann.search(query_vector, top_k)
        
        # Get candidates
        candidates = []
        for i in range(top_k):
            if indices[0, i] == -1:
                continue
            row = self.rows_data[indices[0, i]]
            candidates.append({
                "imagen": row["imagen"],
                "caption": row["texto"],
                "score": distances[0, i]
            })
        
        # Find best result
        if not candidates:
            best_result = None
            image_caption = "No relevant image found."
        else:
            best_idx = self.re_rank_candidates(query_text, candidates)
            best_result = candidates[best_idx]
            image_caption = best_result["caption"]
        
        # Get manual context
        model_response = self.qa_chain({"query": query_text})
        manual_context = model_response.get("result", "No relevant context found.")
        
        # Generate integrated response
        prompt = f"""
You are a vehicle expert virtual assistant. Your responses must be extremely concise and structured.

IMPORTANT GUIDELINES:
1. Use bullet points or numbered lists
2. Keep each point to 1-2 sentences maximum
3. Focus only on essential information
4. Avoid unnecessary explanations
5. Be direct and actionable

Image Description:
{image_caption}

Manual Extract:
{manual_context}

Query:
{query_text}

Provide a structured, concise response following the guidelines above.
"""
        response_generated = self.llm(prompt)
        response = response_generated.content if hasattr(response_generated, 'content') else response_generated
        
        return best_result, response
    
    def create_gui(self):
        """Create and launch the GUI interface"""
        
        def search():
            """Handle search button click"""
            query_text = text_entry.get()
            if not query_text.strip():
                caption_label.config(text="Please enter a query.")
                return
            
            search_button.config(state=tk.DISABLED)
            caption_label.config(text="Searching...")
            context_text.config(state='normal')
            context_text.delete(1.0, tk.END)
            context_text.config(state='disabled')
            image_label.config(image="")
            window.update_idletasks()
            
            def perform_search():
                """Execute search in background thread"""
                try:
                    best_result, response = self.search_by_text(query_text)
                    
                    if best_result and best_result.get("imagen"):
                        image_bytes = best_result["imagen"]
                        image = Image.open(io.BytesIO(image_bytes))
                        image.thumbnail((400, 400))
                        tk_image = ImageTk.PhotoImage(image)
                        image_label.configure(image=tk_image)
                        image_label.image = tk_image
                        caption_label.config(text=best_result["caption"])
                    else:
                        image_label.config(image="")
                        caption_label.config(text="No relevant image found.")
                    
                    context_text.configure(state='normal')
                    context_text.delete(1.0, tk.END)
                    context_text.insert(tk.END, response)
                    context_text.configure(state='disabled')
                except Exception as e:
                    caption_label.config(text=f"Error during search: {e}")
                finally:
                    search_button.config(state=tk.NORMAL)
            
            search_thread = threading.Thread(target=perform_search)
            search_thread.start()
        
        # Create main window
        window = tk.Tk()
        window.title("Multimodal Vehicle Search")
        window.geometry("700x800")
        
        # Create GUI elements
        query_label = ttk.Label(window, text="Enter search query:")
        query_label.pack(pady=10)
        
        text_entry = ttk.Entry(window, width=60)
        text_entry.pack(pady=10)
        
        search_button = ttk.Button(window, text="Search", command=search)
        search_button.pack(pady=10)
        
        image_label = ttk.Label(window)
        image_label.pack(pady=10)
        
        caption_label = ttk.Label(window, text="", wraplength=600)
        caption_label.pack(pady=10)
        
        context_frame = ttk.Frame(window)
        context_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(context_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        context_text = tk.Text(context_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, width=80, height=15)
        context_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=context_text.yview)
        context_text.configure(state='disabled')
        
        window.mainloop()


if __name__ == "__main__":
    app = VehicleAssistantApp()
    app.create_gui()