import os
import io
from pathlib import Path
import torch
import lancedb
import pyarrow as pa
from PIL import Image
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
import json

def cargar_imagenes_y_textos(carpeta_imagenes):
    """
    Recorre cada subcarpeta dentro de la carpeta principal.
    En cada subcarpeta se espera encontrar:
      - Una imagen (archivo .jpg, .jpeg o .png).
      - Uno o más archivos .txt: uno es el caption principal y los demás son Q&A.
        Se intenta identificar el archivo de caption buscando uno que tenga el mismo nombre que la subcarpeta
        o que incluya la palabra "caption". Si no se identifica, se toma el primer archivo .txt como caption.
    Por cada subcarpeta, se generan tantas entradas como archivos de texto se encuentren.
    Retorna dos listas: una con imágenes (repetida para cada texto) y otra con los textos asociados.
    """
    print("Cargando imágenes y textos desde subcarpetas...")
    imagenes_all = []
    textos_all = []
    main_folder = Path(carpeta_imagenes)
    
    for subfolder in main_folder.iterdir():
        if subfolder.is_dir():
            # Buscar la imagen dentro de la subcarpeta
            image_file = None
            for file in subfolder.iterdir():
                if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_file = file
                    break
            if image_file is None:
                print(f"No se encontró imagen en {subfolder.name}.")
                continue
            try:
                imagen = Image.open(image_file).convert("RGB")
            except Exception as e:
                print(f"Error al cargar la imagen {image_file}: {e}")
                continue
            
            # Obtener todos los archivos de texto (.txt) en la subcarpeta
            txt_files = [file for file in subfolder.iterdir() if file.suffix.lower() == ".txt"]
            if not txt_files:
                print(f"No se encontraron archivos de texto en {subfolder.name}.")
                continue
            
            # Identificar el archivo de caption:
            caption_file = None
            # Primero, buscar un archivo cuyo nombre (sin extensión) coincida con el de la subcarpeta.
            for file in txt_files:
                if file.stem.lower() == subfolder.name.lower():
                    caption_file = file
                    break
            # Si no se encontró, buscar alguno que contenga "caption" en su nombre.
            if caption_file is None:
                for file in txt_files:
                    if "caption" in file.stem.lower():
                        caption_file = file
                        break
            # Si aún no se identifica, tomar el primer archivo de la lista.
            if caption_file is None:
                caption_file = txt_files[0]
            
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
            except Exception as e:
                print(f"Error al leer el caption en {caption_file}: {e}")
                caption_text = ""
            
            # Los archivos de Q&A son los demás archivos .txt, excluyendo el caption_file.
            qa_files = [file for file in txt_files if file != caption_file]
            qa_texts = []
            # Ordenamos los archivos para mantener el orden (por ejemplo, Pregunta1.txt, Pregunta2.txt, ...)
            for file in sorted(qa_files):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        qa_text = f.read().strip()
                    if qa_text:
                        qa_texts.append(qa_text)
                except Exception as e:
                    print(f"Error al leer {file}: {e}")
                    continue
            
            # Si no se encontraron Q&A, se toma solo el caption.
            if not qa_texts:
                textos_imagen = [caption_text]
            else:
                textos_imagen = [caption_text] + qa_texts
            
            # Para cada texto (caption y cada Q&A) se añade la misma imagen y el texto correspondiente.
            for texto in textos_imagen:
                imagenes_all.append(imagen)
                textos_all.append(texto)
    
    print("Carga finalizada.")
    return imagenes_all, textos_all

def generar_embeddings(imagenes, textos, processor, model, device, batch_size=8):
    """
    Genera los embeddings a partir de listas de imágenes y textos usando BridgeTower.
    Normaliza y promedia los embeddings de imagen y texto.
    """
    print("Generando Embeddings")
    embeddings = []
    num_samples = len(imagenes)
    embedding_dim = None

    for i in range(0, num_samples, batch_size):
        batch_images = imagenes[i:i+batch_size]
        batch_textos = textos[i:i+batch_size]

        encoding = processor(images=batch_images, text=batch_textos, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalizar embeddings
            image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1)
            text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)

            # Promediar los embeddings de imagen y texto
            combined_embeds = (image_embeds + text_embeds) / 2.0

            if embedding_dim is None:
                embedding_dim = combined_embeds.shape[1]

            combined_embeds = combined_embeds.cpu().numpy()
            embeddings.extend(combined_embeds)

        del encoding, outputs, image_embeds, text_embeds, combined_embeds
        torch.cuda.empty_cache()

    print("Embeddings generados")
    return embeddings, embedding_dim

def insertar_en_lancedb(embeddings, imagenes, textos, lancedb_path, embedding_dim):
    """
    Inserta los embeddings junto con las imágenes y textos en LanceDB.
    """
    print("Guardando embeddings en LanceDB")
    db = lancedb.connect(lancedb_path)

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
        pa.field("imagen", pa.binary()),
        pa.field("texto", pa.string())
    ])

    data = []
    for embedding, imagen, texto in zip(embeddings, imagenes, textos):
        # Convertir imagen a bytes
        with io.BytesIO() as output:
            imagen.save(output, format="PNG")
            imagen_bytes = output.getvalue()

        data.append({
            "vector": embedding.astype('float32').flatten().tolist(),
            "imagen": imagen_bytes,
            "texto": texto
        })
    
    db.create_table("embeddings", data=data, schema=schema, mode='overwrite')
    print("Guardado finalizado")

if __name__ == "__main__":
    # Ruta a la carpeta principal "Manual"
    carpeta_imagenes = r'C:\Users\krato\Documents\Documentos Uni\3° Semestre\Proyecto\Manual'
    lancedb_path = r'C:\Users\krato\Documents\Documentos Uni\3° Semestre\Proyecto\LanceDB_New'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando el dispositivo: {device}")

    # Cargar modelo y procesador de BridgeTower
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc").to(device)

    # Cargar imágenes y textos (caption + Q&A) recorriendo cada subcarpeta de "Manual"
    imagenes, textos = cargar_imagenes_y_textos(carpeta_imagenes)

    # Generar embeddings
    embeddings, embedding_dim = generar_embeddings(imagenes, textos, processor, model, device, batch_size=8)

    # Insertar embeddings en LanceDB
    insertar_en_lancedb(embeddings, imagenes, textos, lancedb_path, embedding_dim)
