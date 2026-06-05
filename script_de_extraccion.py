import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import concurrent.futures
import re
from tqdm import tqdm

sys.path.append('..')
from src.prompter import Prompter

# ── LLM config ──────────────────────────────────────────────────────────────
MODEL_TYPE   = "qwen3:32b"
CONFIG_PATH  = "config.yaml"

# ── Query parameters ────────────────────────────────────────────────────────
ROOT_TOPIC   = "living lab"   # Starting topic
N_KEYWORDS   = 10             # Keywords requested per query
N_KEYWORDS_2 = 50             # Keywords requested per query for levels > 1
MAX_LEVELS   = 5              # How many levels deep to expand (0 = root only)

# ── Cache config ────────────────────────────────────────────────────────────
BASE_DIR = Path("/export/usuarios_ml4ds/danibacaicoa/Living_labs_keyword_extraction_and_analysis")
CACHE_DIR = BASE_DIR / "query_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Initialise prompter ─────────────────────────────────────────────────────
prompter = Prompter(config_path=CONFIG_PATH, model_type=MODEL_TYPE, temperature=0.3)

def _cache_key(topic: str, n_keywords: int) -> str:
    """Stable and readable filename key for a (topic, n_keywords) pair."""
    # Genera nombres de fichero legibles (ej: "focus_groups_n50")
    safe_topic = re.sub(r'[^a-zA-Z0-9_\-]', '_', topic.strip().lower())
    return f"{safe_topic}_n{n_keywords}"

def cache_path(topic: str, n_keywords: int) -> Path:
    return CACHE_DIR / f"{_cache_key(topic, n_keywords)}.json"

def load_cache(topic: str, n_keywords: int) -> dict | None:
    """Return cached result dict or None if not found."""
    p = cache_path(topic, n_keywords)
    if p.exists():
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

def save_cache(topic: str, n_keywords: int, raw_response: str, parsed: dict) -> None:
    record = {
        "topic": topic,
        "n_keywords": n_keywords,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw_response,
        "parsed": parsed,
    }
    with open(cache_path(topic, n_keywords), "w", encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

def invalidate_cache(topic: str, n_keywords: int) -> None:
    """Delete a cached entry to force a fresh LLM call."""
    p = cache_path(topic, n_keywords)
    if p.exists():
        p.unlink()
        print(f"Cache cleared for '{topic}'.")
    else:
        print(f"No cache entry found for '{topic}'.")

print("Cache helpers loaded. Cache directory:", CACHE_DIR.resolve())

DOMAIN_DEFINITIONS = {
    "living lab": {
        "full": (
            "A Living Lab is a real-world open innovation ecosystem where citizens, researchers, "
            "companies, and governments co-create and test solutions in everyday life contexts."
        ),
        "dimensions": [
            "real-world testing and experimentation contexts",
            "open innovation processes and methodologies",
            "multi-actor collaboration (citizens, researchers, companies, governments)",
            "co-creation and participatory design methodologies",
            "everyday life application and urban/social contexts",
        ]
    }
}

def build_prompt(topic: str, n_keywords: int, root_topic: str = "") -> str:
    is_root = not root_topic or topic == root_topic
    domain_def = DOMAIN_DEFINITIONS.get(topic if is_root else root_topic, None)

    if domain_def:
        if is_root:
            definition_block = f"\nTOPIC DEFINITION: {domain_def['full']}\n"
        else:
            dims = "\n".join(f"  - {d}" for d in domain_def["dimensions"])
            definition_block = f"""
                SCORING CRITERIA: A keyword scores 8-10 only if it directly relates to at least 
                one of these dimensions:
                {dims}
                Keywords unrelated to these dimensions should score 1-3 at most.
                """
    else:
        definition_block = ""

    return f"""You are a JSON-only output machine. You never write explanations, greetings, or markdown.
                TASK: Given the topic "{topic}", return exactly {n_keywords} related keywords with specificity scores.
                {definition_block}
                RULES (violations will break the system):
                1. Output MUST start with {{ and end with }} — nothing before, nothing after
                2. The topic itself must appear first with score 0
                3. No keyword may contain the topic word "{topic}"
                4. All keywords must be unique
                5. Scores are integers 1-10 only (topic gets 0)
                6. No markdown, no ```json, no explanations, no trailing text
                SCORING:
                8-10 → direct synonyms or highly specific sub-concepts
                4-7  → broadly related fields or associated concepts  
                1-3  → loose umbrella terms
                TOPIC: {topic}
                OUTPUT (raw JSON only):
                {{
                "{topic}": 0,
                "keyword1": score,
                "keyword2": score
                }}"""

def extract_dictionary(response: str) -> dict:
    """Strip optional markdown fences and parse JSON, handling LLM reasoning errors."""
    # 1. Eliminar posible bloque <think>
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # 2. ELIMINAR CARACTERES DE CONTROL INVISIBLES (como el \x1f) QUE ROMPEN EL JSON
    clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean)
    
    # 3. Forzar la extracción solo del interior de las llaves {} (ignorar texto suelto alrededor)
    match = re.search(r'\{.*\}', clean, flags=re.DOTALL)
    if match:
        clean = match.group(0)
        
    return json.loads(clean.strip())

def get_keywords_with_position(topic: str, n_keywords: int, prompter,
                               root_topic: str = "",
                               use_cache: bool = True, max_retries: int = 3) -> dict:
    """
    Versión silenciosa: Extrae la keyword, filtra las inválidas sin lanzar error 
    y guarda la posición.
    """
    if use_cache:
        cached = load_cache(topic, n_keywords)
        if cached is not None:
            return cached["parsed"]

    last_error = None
    raw = None
    
    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_prompt(topic, n_keywords, root_topic=root_topic)
            print(f"Consultando '{topic}' (n={n_keywords})...")
            
            raw = prompter.prompt(question=prompt, system_prompt_template_path=None)[0]
            parsed = extract_dictionary(raw)
            
            result = {}
            position = 0
            topic_lower = topic.strip().lower()
            
            # Aseguramos de que el topic original esté insertado en 0
            result[topic] = {"score": 0, "position": 0}
            
            # Filtrado inteligente
            for kwd, score in parsed.items():
                kwd_clean = kwd.strip()
                kwd_lower = kwd_clean.lower()
                
                # Ignorar si es idéntico al topic 
                if kwd_lower == topic_lower:
                    continue
                    
                # Ignorar si el score no es un entero en el rango esperado
                if not isinstance(score, int) or not (1 <= score <= 10):
                    continue
                
                # SÍ PERMITIMOS palabras que contengan al topic (ej. "systems thinking tools")
                position += 1
                result[kwd_clean] = {"score": score, "position": position}
            
            if len(result) < 2:
                raise ValueError("Faltan keywords válidas tras el filtrado.")
                
            save_cache(topic, n_keywords, raw, result)
            print(f"  ✓ {position} términos obtenidos y limpiados exitosamente")
            return result

        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{max_retries} failed for '{topic}': {e}")
            time.sleep(2 * attempt)

    # Si falla por completo, guarda un registro pero devuelve un dict inofensivo 
    # para que la ejecución en paralelo de los otros topics no se aborte.
    if raw:
        bad_path = CACHE_DIR / f"FAILED__{_cache_key(topic, n_keywords)}.txt"
        bad_path.write_text(raw, encoding='utf-8')
        print(f"Fallo definitivo para '{topic}'. Omitiendo este nodo para continuar.")
        
    return {topic: {"score": 0, "position": 0}}

def expand_keywords_parallel(keywords_list: list, lvl: int, root_topic: str, prompter):
    """Expande múltiples keywords en paralelo usando get_keywords_with_position."""
    if not keywords_list:
        return []
    
    print(f"  Expandiendo {len(keywords_list)} nodos en paralelo (nivel {lvl})...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_kwd = {
            executor.submit(get_keywords_with_position, kwd, N_KEYWORDS_2, prompter, root_topic): kwd
            for kwd in keywords_list
        }
        for future in concurrent.futures.as_completed(future_to_kwd):
            kwd = future_to_kwd[future]
            try:
                data = future.result()
                results.append((kwd, data))
            except Exception as exc:
                print(f" '{kwd}' falló silenciosamente: {exc}")
    
    print(f" Completados {len(results)}/{len(keywords_list)} nodos del bloque")
    return results

# ============================================================================
# INICIO DE RECOLECCIÓN Y CACHÉ
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("INICIANDO EXTRACCIÓN AL CACHÉ POR NIVELES (CRAWLER)")
    print("="*60)

    # Registro de todas las keywords exploradas para EVITAR duplicados 
    # en cualquier nivel o entre diferentes ramas.
    all_explored_nodes = set([ROOT_TOPIC])

    # 1. Nivel raíz
    print("\nPASO 1: Procesando nodo raíz...")
    root_data = get_keywords_with_position(ROOT_TOPIC, N_KEYWORDS, prompter, ROOT_TOPIC)
    print(f"  Raíz expandida: {len(root_data)-1} términos directos")

    # 2. Preparar el Nivel 1
    current_level_nodes = [kwd for kwd in root_data.keys() if kwd != ROOT_TOPIC]
    all_explored_nodes.update(current_level_nodes)

    # 3. Expansión por el resto de niveles
    for lvl in range(1, MAX_LEVELS):
        print(f"\nPASO {lvl+1}: Expandiendo nivel {lvl} ({len(current_level_nodes)} nodos)...")
        
        # Expansión paralela (LLM y guardado auto en archivos json)
        expansion_results = expand_keywords_parallel(current_level_nodes, lvl, ROOT_TOPIC, prompter)
        
        next_level_nodes = []
        
        for parent_kwd, data in expansion_results:
            for child_kwd in data.keys():
                # Ignorar keywords inválidas, la propia raíz, o el mismo padre
                if child_kwd == parent_kwd or child_kwd == ROOT_TOPIC:
                    continue
                    
                # Añadir solo si es una palabra totalmente nueva en toda la ejecución
                if child_kwd not in all_explored_nodes:
                    next_level_nodes.append(child_kwd)
                    all_explored_nodes.add(child_kwd)
        
        # Siguiente generación
        current_level_nodes = list(set(next_level_nodes))
        print(f"  ✅ Nivel {lvl} completado: {len(current_level_nodes)} nodos únicos y no repetidos pasados al siguiente nivel.")

    print("\n✅ ¡FASE DE CRAWLING Y CACHÉ COMPLETADA EXITOSAMENTE!")
    print(f"Todos los archivos JSON limpios están listos en: {CACHE_DIR.resolve()}")