import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.documents import Document
import warnings
import json
from typing import List, Dict, Tuple, Optional
import time
from functools import lru_cache
import urllib3
import ssl
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

# Configuration des logs
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'llogyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Constantes pour la configuration d'Ollama
OLLAMA_HOST = "http://127.0.0.1:11434"  # Adresse du serveur Ollama local
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"  # User-Agent pour les requêtes HTTP
PERSIST_DIRECTORY = "contests"  # Répertoire de persistance pour la base vectorielle
COLLECTION_NAME = "contests"  # Nom de la collection dans la base vectorielle

# Configuration des modèles d'IA
LLM_MODEL = "gemma3:latest"  # Modèle de langage pour le traitement du texte
# nomic-embed-text znbang/bge:small-en-v1.5-q8_0 mxbai-embed-large
EMBEDDING_MODEL = "mxbai-embed-large"  # Modèle pour la génération des embeddings

# Paramètres de configuration du LLM
LLM_TEMPERATURE = 0.0  # Contrôle de la créativité (0 = plus déterministe)
LLM_TOP_P = 0.5  # Seuil de probabilité pour le sampling
LLM_NUM_CTX = 4096  # Taille du contexte en tokens

# Paramètres pour le découpage du texte
CHUNK_SIZE = 768  # Taille des segments de texte en caractères
CHUNK_OVERLAP = 200  # Chevauchement entre les segments
MIN_TEXT_LENGTH = 50  # Longueur minimale pour considérer un texte

# Paramètres de traitement par lots
BATCH_SIZE = 32  # Nombre d'éléments traités simultanément
SEARCH_TOP_K = 4  # Nombre de résultats de recherche à retourner
LIST_TOP_K = 500  # Nombre maximum d'éléments à lister

# Paramètres pour la base vectorielle HNSW
HNSW_SPACE = "cosine"  # Métrique de similarité utilisée
HNSW_EF = 100  # Facteur d'exploration pour la recherche

# Configuration de l'environnement
os.environ["USER_AGENT"] = USER_AGENT
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

# Désactiver les avertissements SSL
warnings.filterwarnings("ignore", category=DeprecationWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration SSL permissive
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration du cache global
set_llm_cache(InMemoryCache())

class EnhancedWebLoader(WebBaseLoader):
    """Loader personnalisé pour extraire les liens et images avec leur contexte"""
    
    def _get_link_context(self, link, soup) -> str:
        """Extrait le contexte d'un lien"""
        context = []
        
        # Récupérer le texte du lien
        link_text = link.get_text(strip=True)
        if link_text:
            context.append(f"Texte du lien: {link_text}")
        
        # Récupérer l'URL
        href = link.get('href')
        if href:
            absolute_url = urljoin(self.web_path, href)
            context.append(f"URL: {absolute_url}")
        
        # Récupérer le paragraphe parent
        parent = link.find_parent(['p', 'div', 'section'])
        if parent:
            parent_text = parent.get_text(strip=True)
            if len(parent_text) > 500:  # Limiter la taille du contexte
                parent_text = parent_text[:500] + "..."
            context.append(f"Contexte: {parent_text}")
        
        return " | ".join(context)
    
    def _get_image_context(self, img, soup) -> str:
        """Extrait le contexte d'une image"""
        context = []
        
        # Récupérer l'alt text
        alt = img.get('alt', '')
        if alt:
            context.append(f"Description: {alt}")
        
        # Récupérer le titre
        title = img.get('title', '')
        if title:
            context.append(f"Titre: {title}")
        
        # Récupérer l'URL
        src = img.get('src')
        if src:
            absolute_url = urljoin(self.web_path, src)
            context.append(f"URL: {absolute_url}")
        
        # Récupérer la légende si elle existe
        figcaption = img.find_parent('figure').find('figcaption') if img.find_parent('figure') else None
        if figcaption:
            context.append(f"Légende: {figcaption.get_text(strip=True)}")
        
        return " | ".join(context)
    
    def load(self) -> List[Document]:
        """Charge la page et extrait le contenu avec les liens et images"""
        logger.info(f"Chargement de la page {self.web_path}")
        
        # Configuration de Chrome en mode headless
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument(f'user-agent={os.environ["USER_AGENT"]}')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Charger la page
            driver.get(self.web_path)
            
            # Attendre que la page soit chargée (attendre que le body soit présent)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Faire défiler la page jusqu'en bas
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                # Défiler jusqu'en bas
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Attendre le chargement du contenu
                time.sleep(2)
                
                # Calculer la nouvelle hauteur
                new_height = driver.execute_script("return document.body.scrollHeight")
                
                # Si la hauteur n'a pas changé, on a atteint le bas
                if new_height == last_height:
                    break
                last_height = new_height
            
            # Récupérer le contenu de la page après le chargement complet
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            documents = []
            
            # Extraire les liens avec leur contexte
            links = soup.find_all('a', href=True)
            logger.debug(f"Extraction de {len(links)} liens")
            for link in links:
                link_info = self._get_link_context(link, soup)
                if link_info:
                    documents.append(
                        Document(
                            page_content=f"[LIEN] {link_info}",
                            metadata={
                                "source": self.web_path,
                                "type": "link"
                            }
                        )
                    )
            
            # Extraire les images avec leur contexte
            images = soup.find_all('img')
            logger.debug(f"Extraction de {len(images)} images")
            for img in images:
                img_info = self._get_image_context(img, soup)
                if img_info:
                    documents.append(
                        Document(
                            page_content=f"[IMAGE] {img_info}",
                            metadata={
                                "source": self.web_path,
                                "type": "image"
                            }
                        )
                    )
            
            # Extraire le contenu textuel principal
            text_count = 0
            for text in soup.stripped_strings:
                if len(text.strip()) > MIN_TEXT_LENGTH:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": self.web_path,
                                "type": "text"
                            }
                        )
                    )
                    text_count += 1
            
            logger.debug(f"Extraction de {text_count} blocs de texte")
            logger.info(f"Total de {len(documents)} documents extraits")
            return documents
            
        finally:
            # Fermer le navigateur
            driver.quit()

# Cache pour les instances LLM et embeddings
@lru_cache(maxsize=1)
def get_llm():
    logger.debug("Initialisation du LLM")
    return Ollama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        num_ctx=LLM_NUM_CTX,
    )

@lru_cache(maxsize=1)
def get_embeddings():
    logger.debug("Initialisation des embeddings")
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST
    )

# Initialisation de la base vectorielle avec configuration optimisée
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=get_embeddings(),
    collection_name=COLLECTION_NAME,
    collection_metadata={"hnsw:space": HNSW_SPACE, "hnsw:construction_ef": HNSW_EF}
    )

def add_document(document_path: str) -> bool:
    """
    Ajoute un document s'il n'existe pas déjà dans la base.
    
    Args:
        document_path: URL ou chemin vers le document à ajouter
        
    Returns:
        bool: True si le document a été ajouté, False s'il existait déjà
    """
    try:
        logger.info(f"Tentative d'ajout du document: {document_path}")
        
        # Vérifier si le document existe déjà avec cache
        results = vectorstore.similarity_search_with_score(
            "dummy query",
            k=1,
            filter={"source": document_path}
        )
        
        if len(results) > 0:
            logger.info(f"Le document '{document_path}' est déjà dans la base.")
            return False
            
        # Charger et découper le document avec configuration optimisée
        loader = EnhancedWebLoader(
            document_path,
            verify_ssl=False,  # Désactiver la vérification SSL
        )
        documents = loader.load()
        
        logger.debug(f"Découpage des documents en chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        
        splits = text_splitter.split_documents(documents)
        logger.debug(f"Nombre de chunks créés: {len(splits)}")
        
        # Ajouter à la base vectorielle avec batch processing
        total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            logger.debug(f"Traitement du batch {batch_num}/{total_batches}")
            vectorstore.add_documents(batch)
            
        logger.info(f"Document {document_path} ajouté avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout du document: {str(e)}")
        logger.debug(f"Détails de l'erreur:", exc_info=True)
        return False

def list_documents():
    """Liste tous les documents dans la base avec optimisation."""
    logger.info("Listage des documents dans la base")
    results = vectorstore.similarity_search_with_score(
        "dummy query", 
        k=LIST_TOP_K
    )
    
    # Dictionnaire pour stocker les statistiques par URL
    doc_stats = {}
    
    for doc, _ in results:
        if 'source' in doc.metadata:
            url = doc.metadata['source']
            if url not in doc_stats:
                doc_stats[url] = {
                    'chunks': 0,
                    'total_size': 0
                }
            doc_stats[url]['chunks'] += 1
            doc_stats[url]['total_size'] += len(doc.page_content)
    
    # Afficher les statistiques pour chaque document
    for url, stats in doc_stats.items():
        print(f"Document: {url}")
        print(f"  Nombre de chunks: {stats['chunks']}")
        print(f"  Taille totale: {stats['total_size']} caractères")
        print("---")
    
    logger.debug(f"Nombre de documents uniques trouvés: {len(doc_stats)}")

def query_documents(question: str) -> Tuple[str, List[Dict]]:
    """
    Interroge la base de documents avec optimisation.
    
    Args:
        question: La question à poser
    
    Returns:
        tuple: (réponse, sources)
    """
    logger.debug(f"Traitement de la requête: {question}")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": SEARCH_TOP_K}
        ),
        return_source_documents=True
    )
    
    logger.debug("Exécution de la chaîne QA")
    result = qa_chain({"query": question})
    
    sources = []
    for doc in result["source_documents"]:
        sources.append({
            "url": doc.metadata.get("source", ""),
            "content": doc.page_content
        })
    
    logger.debug(f"Nombre de sources trouvées: {len(sources)}")
    return result["result"], sources

def close_vectorstore():
    """Ferme proprement la base vectorielle et persiste les données."""
    global vectorstore
    
    if vectorstore:
        logger.info("Fermeture de la base vectorielle")
        try:
            # Persister les changements
            vectorstore.persist()
            # Libérer l'instance
            vectorstore = None
            logger.info("Base vectorielle fermée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la base: {e}")
            logger.debug("Détails de l'erreur:", exc_info=True)

def open_vectorstore():
    """Ouvre la base vectorielle existante."""
    global vectorstore
    
    if not vectorstore:
        logger.info("Ouverture de la base vectorielle")
        try:
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=get_embeddings(),
                collection_name=COLLECTION_NAME,
                collection_metadata={"hnsw:space": HNSW_SPACE, "hnsw:construction_ef": HNSW_EF},
            )
            logger.info("Base vectorielle ouverte avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'ouverture de la base: {e}")
            logger.debug("Détails de l'erreur:", exc_info=True)
            raise

# Test du script
if __name__ == "__main__":
    start_time = time.time()
    logger.debug("Démarrage du script")
    
    # Rouvrir la base pour les requêtes
    open_vectorstore()
    
    # Ajout des documents
    add_document("https://ndawards.net/")
    add_document("https://photocontestguru.com/contests/nd-awards-2025/")
    add_document("https://www.photocontestguru.com/contests/nd-awards-2025/")
    add_document("https://www.poesie-francaise.fr/francois-coppee/poeme-pour-toujours.php")
    add_document("https://poetesenberry.over-blog.com/2018/06/le-poeme-le-plus-long-du-monde.html")
    
    # Liste des documents
    list_documents()
    
    # Test de requête
    question = """
GOALS :
Give the best URL of the terms and conditions to participate this contest: "ND Awards 2025" and only this contest.

** IMPORTANT:
- if you don't know the answer, don't respond.
- Give only the URL answer without header, without comments, without explanations, nothing else than the URL.

    """    
    answer, sources = query_documents(question)
    print("\n\nURL TAC du concours : ", answer)
    #print("\n\nSOURCES : ", json.dumps(sources, indent=4))
    
        # Test de requête
    question = """
GOALS :
Give the best URL of the image representing contest: "ND Awards 2025".

** IMPORTANT:
- if you don't know the answer, don't respond.
- Give only the URL answer without header, without comments, without explanations, nothing else than the URL.

    """    
    answer, sources = query_documents(question)
    print("\n\nImage du concours : ", answer)
    #print("\n\nSOURCES : ", json.dumps(sources, indent=4))

    # Fermer la base pour persister les données
    close_vectorstore()

    
    execution_time = time.time() - start_time
    logger.debug(f"Temps d'exécution total: {execution_time:.2f} secondes")
