import os
import sys

# Ensure Python can find the trace_memory package in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from datetime import datetime
import base64
import threading
import requests
from typing import List, Dict, Any, Optional
import openai
from trace_memory import CTree, TopicNode, MessageNode
from trace_memory._llm_utils import ChatGPT_API, extract_json
from trace_memory import VectorDatabase, ConversationVector
from trace_memory import PromptSynthesizer

# Native .env loader so users don't need python-dotenv installed
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

# Use standard OpenAI endpoint format. If users provide a base URL in .env, they should include /v1
LM_STUDIO_URL = os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:1234/v1')
LM_STUDIO_API_KEY = os.getenv('OPENAI_API_KEY', 'lm-studio')

# Allow separate endpoint for vision (e.g., if using local LM Studio for vision but NVIDIA API for text)
VISION_BASE_URL = os.getenv('NEXUS_VISION_BASE_URL', LM_STUDIO_URL)
VISION_API_KEY = os.getenv('NEXUS_VISION_API_KEY', LM_STUDIO_API_KEY)

EMBEDDING_BASE_URL = os.getenv('NEXUS_EMBEDDING_BASE_URL', LM_STUDIO_URL)
EMBEDDING_API_KEY = os.getenv('NEXUS_EMBEDDING_API_KEY', LM_STUDIO_API_KEY)

TEXT_MODEL = os.getenv('NEXUS_TEXT_MODEL', 'meta-llama-3.1-8b-instruct-abliterated')
VISION_MODEL = os.getenv('NEXUS_VISION_MODEL', 'moondream-2b-2025-04-14')
EMBEDDING_MODEL = os.getenv('NEXUS_EMBEDDING_MODEL', 'text-embedding-nomic-embed-text-v1.5')
_active_model = TEXT_MODEL
_model_lock = threading.Lock()
SAVED_CHATS_DIR = 'saved_chats'
INGEST_DIR = '_ingest'
os.makedirs(SAVED_CHATS_DIR, exist_ok=True)
os.makedirs(INGEST_DIR, exist_ok=True)
CLEAR_SCREEN = '\x1b[2J\x1b[H'
RESET = '\x1b[0m'
BOLD = '\x1b[1m'
DIM = '\x1b[2m'
COLOR_ACCENT = '\x1b[38;5;99m'
COLOR_USER = '\x1b[38;5;81m'
COLOR_AI = '\x1b[38;5;121m'
COLOR_SYSTEM = '\x1b[38;5;214m'
COLOR_SLATE = '\x1b[38;5;244m'
COLOR_SUCCESS = '\x1b[38;5;46m'
COLOR_WARNING = '\x1b[38;5;196m'
COLOR_MODEL = '\x1b[38;5;220m'

class Spinner:
    CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, message='Thinking...', delay=0.08):
        self._message = message
        self._delay = delay
        self._stop_event = threading.Event()
        self._thread = None
        self._msg_lock = threading.Lock()

    @property
    def message(self):
        with self._msg_lock:
            return self._message

    @message.setter
    def message(self, value):
        with self._msg_lock:
            self._message = value

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            with self._msg_lock:
                msg = self._message
            line = f'\r{COLOR_ACCENT}{self.CHARS[idx]} {COLOR_SLATE}{msg}{RESET}'
            sys.stdout.write(line)
            sys.stdout.flush()
            idx = (idx + 1) % len(self.CHARS)
            time.sleep(self._delay)
        with self._msg_lock:
            clear_width = len(self._message) + 4
        sys.stdout.write('\r' + ' ' * clear_width + '\r')
        sys.stdout.flush()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None

def get_openai_client():
    return openai.OpenAI(api_key=LM_STUDIO_API_KEY, base_url=LM_STUDIO_URL)

def get_vision_client():
    return openai.OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)

def get_embedding_client():
    return openai.OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

def generate_completion(messages, temperature=0.7, max_tokens=1500, model=None):
    global _active_model
    client = get_openai_client()
    use_model = model or TEXT_MODEL
    try:
        response = client.chat.completions.create(model=use_model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        content = response.choices[0].message.content
        return content if content is not None else ""
    except Exception as e:
        return f'{COLOR_WARNING}[LM Studio Error] {e}{RESET}'

_fallback_embedder = None

def _get_fallback_embedding(text):
    global _fallback_embedder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("Embedding API failed, and 'sentence-transformers' is not installed for fallback. Please install it (e.g., pip install sentence-transformers).")
    
    if _fallback_embedder is None:
        print(f"\n{COLOR_WARNING}[Fallback] Loading local embedding model (BAAI/bge-base-en-v1.5)...{RESET}")
        # BAAI/bge-base-en-v1.5 is a highly rated embedding model
        _fallback_embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # encode returns a numpy array, convert to list of floats
    return _fallback_embedder.encode(text).tolist()

def get_embedding(text, model=EMBEDDING_MODEL):
    client = get_embedding_client()
    cleaned = text.replace('\n', ' ')
    try:
        response = client.embeddings.create(input=[cleaned], model=model)
        return response.data[0].embedding
    except Exception as e:
        try:
            return _get_fallback_embedding(cleaned)
        except Exception as fallback_e:
            raise RuntimeError(f'Failed to generate embedding via API ({e}) and fallback also failed: {fallback_e}') from fallback_e

def _get_loaded_model_ids():
    try:
        resp = requests.get(f'{LM_STUDIO_URL}/models', timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ids = []
            if 'data' in data:
                for m in data['data']:
                    ids.append(m['id'])
            return ids
    except Exception:
        pass
    return []

def _is_model_loaded(model_id):
    loaded = _get_loaded_model_ids()
    model_id_lower = model_id.lower()
    return any((model_id_lower in m.lower() or m.lower() in model_id_lower for m in loaded))

def swap_model(target_model_name):
    global _active_model
    MAX_WAIT_SECS = 120
    POLL_INTERVAL = 2.0
    with _model_lock:
        if _active_model == target_model_name:
            return True
    mgmt_base = LM_STUDIO_URL.replace('/v1', '')
    spinner = Spinner(f'[VRAM] Swapping to model: {target_model_name} …')
    spinner.start()
    try:
        try:
            requests.post(f'{mgmt_base}/api/v0/models/unload', json={'model': _active_model}, timeout=10)
        except Exception:
            pass
        time.sleep(1.5)
        try:
            requests.post(f'{mgmt_base}/api/v0/models/load', json={'model': target_model_name}, timeout=10)
        except Exception:
            pass
        elapsed = 0.0
        model_ready = False
        spinner.message = f"[VRAM] Waiting for '{target_model_name}' to initialise on GPU…"
        while elapsed < MAX_WAIT_SECS:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            if _is_model_loaded(target_model_name):
                model_ready = True
                break
            try:
                probe_client = get_openai_client()
                probe_client.chat.completions.create(model=target_model_name, messages=[{'role': 'user', 'content': 'hi'}], max_tokens=1, temperature=0)
                model_ready = True
                break
            except Exception:
                pass
        spinner.stop()
        if model_ready:
            with _model_lock:
                _active_model = target_model_name
            sys.stdout.write(f' {COLOR_MODEL}[VRAM] ✓ Model swapped → {BOLD}{target_model_name}{RESET}\n')
            sys.stdout.flush()
            return True
        else:
            sys.stdout.write(f" {COLOR_WARNING}[VRAM] ✗ Timeout waiting for '{target_model_name}' after {MAX_WAIT_SECS}s. Proceeding anyway.{RESET}\n")
            sys.stdout.flush()
            return False
    except Exception as e:
        spinner.stop()
        sys.stdout.write(f' {COLOR_WARNING}[VRAM] Swap error: {e}{RESET}\n')
        sys.stdout.flush()
        return False


def describe_image_via_vision(image_path):
    ext = os.path.splitext(image_path.lower())[1]
    mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
    try:
        with open(image_path, 'rb') as f:
            b64_image = base64.b64encode(f.read()).decode('utf-8')
    except OSError as e:
        return f'[Vision Error] Could not read image file: {e}'
    sys.stdout.write(f'\n {COLOR_MODEL}[VRAM] Vision ingest detected — swapping to vision model…{RESET}\n')
    sys.stdout.flush()
    swap_model(VISION_MODEL)
    description = '[Vision model did not return a description]'
    try:
        client = get_vision_client()
        response = client.chat.completions.create(model=VISION_MODEL, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe this image in rich detail for a long-term memory retrieval system. Include key subjects, objects, text visible, spatial layout, dominant colours, and any notable actions or emotions. Be thorough.'}, {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{b64_image}'}}]}], max_tokens=512, temperature=0.2)
        description = response.choices[0].message.content.strip()
    except Exception as e:
        description = f'[Vision API Error] {e}. Ensure LM Studio has a vision model (e.g. moondream2) available.'
    finally:
        sys.stdout.write(f' {COLOR_MODEL}[VRAM] Vision complete — swapping back to text model…{RESET}\n')
        sys.stdout.flush()
        swap_model(TEXT_MODEL)
    return description

def search_web(query: str) -> str:
    """Run a DuckDuckGo search and return a compact, readable result block."""
    try:
        from ddgs import DDGS
    except ImportError:
        return '[ddgs not installed — run: pip install ddgs]'
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        if not hits:
            return 'No results found.'
        lines = []
        for h in hits:
            title = h.get('title', '').strip()
            body  = (h.get('body', '') or '').strip()[:300]
            href  = h.get('href', '').strip()
            lines.append(f'• {title}\n  {body}\n  {href}')
        return '\n\n'.join(lines)
    except Exception as exc:
        return f'[Search error: {exc}]'

# ─ Keywords that always signal time-sensitive content ─────────────────────────────────────
_SEARCH_KEYWORDS = [
    'price', 'cost', 'buy', 'cheapest', 'best', 'top', 'latest',
    'current', 'now', 'today', 'recent', 'new', 'release', '2025', '2026',
    'rate', 'score', 'news', 'update', 'launch', 'available', 'offer',
    'deal', 'discount', 'emi', 'lakh', 'rupee', '₹', 'stock', 'vs',
    'compare', 'review', 'weather', 'election', 'winner',
]

_NO_SEARCH_KEYWORDS = [
    'explain', 'what is', 'how does', 'define', 'meaning', 'difference between',
    'write', 'summarize', 'translate', 'fix', 'debug', 'code', 'algorithm',
    'history of', 'who was', 'math', 'calculate', 'solve',
]

def route_search_decision(user_input: str) -> dict:
    """
    Fast pre-generation router: decides whether a web search is needed
    before the main LLM call runs.
    """
    lower = user_input.lower()

    for kw in _NO_SEARCH_KEYWORDS:
        if kw in lower:
            return {'search': False, 'query': ''}

    keyword_hits = sum(1 for kw in _SEARCH_KEYWORDS if kw in lower)
    if keyword_hits >= 2:
        now_year = datetime.now().year
        return {'search': True, 'query': f'{user_input.strip()} {now_year}'}

    router_prompt = (
        f'Today is {datetime.now().strftime("%d %B %Y")}.\n'
        'Your job: decide if the query below needs a live web search to answer accurately.\n\n'
        'Search IS needed for: current prices, news, live data, latest products/versions, '
        'rankings as of today, events after 2024.\n'
        'Search is NOT needed for: math, code, writing, history, definitions, explanations.\n\n'
        f'Query: "{user_input}"\n\n'
        'Reply with ONLY a JSON object, nothing else:\n'
        '{"search": true|false, "query": "optimised search query or empty string"}'
    )
    try:
        raw = generate_completion(
            [{'role': 'user', 'content': router_prompt}],
            temperature=0.0, max_tokens=60, model=TEXT_MODEL
        )
        decision = extract_json(raw)
        if not decision:
            decision = {}
        search = bool(decision.get('search', False))
        query  = str(decision.get('query', '')).strip() or user_input
        if search and not query:
            query = f'{user_input.strip()} {datetime.now().year}'
        return {'search': search, 'query': query}
    except Exception:
        return {'search': False, 'query': ''}

def construct_compacted_memory_context(tree):
    blocks = []
    blocks.append('── GLOBAL CONVERSATION MEMORY INDEX ──')

    def _traverse(node, depth=0):
        lines = []
        indent = '  ' * depth
        if isinstance(node, TopicNode) and node.topic_name != 'ROOT':
            summary = (node.summary or 'Active discussion in progress.').strip()
            lines.append(f'{indent}• {node.topic_name} [msgs {node.start_index}–{node.end_index}]')
            lines.append(f'{indent}  ↳ {summary}')
        for child in node.children:
            if isinstance(child, TopicNode):
                lines.extend(_traverse(child, depth + 1))
        return lines
    summaries = _traverse(tree.root)
    blocks.extend(summaries if summaries else ['  (No topics indexed yet)'])
    blocks.append('\n── ACTIVE THEMATIC CONTEXT PATH ──')
    ancestors = tree.get_ancestors(tree.current_node, include_self=True, exclude_root=True)
    if ancestors:
        path = ' → '.join((n.topic_name for n in ancestors))
        blocks.append(f'Thread: {path}')
        for node in ancestors:
            s = (node.summary or 'Expanding…').strip()
            blocks.append(f'  • {node.topic_name}: {s}')
    else:
        blocks.append('  Thread: ROOT (first topic not yet created)')
    blocks.append('\n── RECENT DETAILED CONVERSATION LOGS ──')
    recent = tree.conversation[-8:] if tree.conversation else []
    if recent:
        for msg in recent:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            if isinstance(content, str) and len(content) > 600:
                content = content[:600] + '… [truncated]'
            elif isinstance(content, list):
                content = '[multimodal message]'
            blocks.append(f'[{role}]: {content}')
    else:
        blocks.append('  (No messages yet)')
    return '\n'.join(blocks)

def background_generate_title_and_rename(tree, first_user_prompt, first_ai_response, current_filepath):
    prompt = f'Based on the conversation below, output a concise 3–4-word title only.\nNo punctuation, quotes, or extra words. Just the title.\n\nUser: {first_user_prompt[:300]}\nAI: {first_ai_response[:300]}'
    try:
        title = generate_completion([{'role': 'user', 'content': prompt}], temperature=0.3, max_tokens=15)
        clean = ''
        for c in title:
            if c.isalnum() or c == ' ' or c == '-' or (c == '_'):
                clean += c
        clean = clean.strip()
        ts = int(time.time())
        new_filename = f'{clean.lower()}_{ts}.json'
        new_filepath = os.path.join(SAVED_CHATS_DIR, new_filename)
        if os.path.exists(current_filepath):
            os.rename(current_filepath, new_filepath)
        tree.auto_save_path = new_filepath
        current_db = current_filepath.replace('.json', '_vectors.db')
        new_db = new_filepath.replace('.json', '_vectors.db')
        if os.path.exists(current_db):
            try:
                os.rename(current_db, new_db)
                if hasattr(tree, 'vdb') and tree.vdb is not None:
                    tree.vdb = VectorDatabase(new_db)
                    if hasattr(tree, 'synthesizer') and tree.synthesizer is not None:
                        tree.synthesizer.vdb = tree.vdb
            except Exception:
                pass
        if tree.auto_save_path:
            tree.save(tree.auto_save_path, save_conversation=True)
    except Exception:
        pass

def _word_wrap(text, width=63):
    lines = []
    current = ''
    for word in text.split():
        while len(word) > width:
            lines.append(word[:width])
            word = word[width:]
        if len(current) + len(word) + (1 if current else 0) <= width:
            current = (current + ' ' + word).lstrip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def background_generate_welcome_back(tree, print_callback):
    top_topics = [c for c in tree.root.children if isinstance(c, TopicNode)]
    snippets = []
    for topic in top_topics:
        if topic.summary:
            snippets.append(f'• {topic.topic_name}: {topic.summary}')
    if not snippets:
        for msg in tree.conversation[-4:]:
            if msg.get('role') == 'system':
                continue
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            if isinstance(content, list):
                content = '[multimodal]'
            snippets.append(f'• {role}: {content[:120]}…')
    summary_block = '\n'.join(snippets) or '• (no history available)'
    prompt = f'You are a memory assistant. Write exactly 2 warm, concise sentences:\n1. Welcome the user back by name-agnostic greeting.\n2. Briefly recall what was discussed, based ONLY on the memories below.\nDo not use placeholders. Be natural.\n\nMemories:\n{summary_block}'
    try:
        msg = generate_completion([{'role': 'user', 'content': prompt}], temperature=0.6, max_tokens=160)
        print_callback(msg.strip())
    except Exception:
        print_callback("Welcome back! I'm ready to continue where we left off.")


def perform_ingest_operation(tree):
    all_files = []
    for f in os.listdir(INGEST_DIR):
        if os.path.isfile(os.path.join(INGEST_DIR, f)):
            all_files.append(f)
    if not all_files:
        return f"{COLOR_WARNING}No files found in '{INGEST_DIR}/'. Drop a .png, .jpg, or .jpeg there first.{RESET}"
    results = []
    for filename in all_files:
        filepath = os.path.join(INGEST_DIR, filename)
        ext = os.path.splitext(filename.lower())[1]
        sys.stdout.write(f'{COLOR_SLATE}  → Processing: {COLOR_ACCENT}{filename}{RESET}\n')
        sys.stdout.flush()
        if ext in ('.png', '.jpg', '.jpeg'):
            try:
                description = describe_image_via_vision(filepath)
                short_desc = description[:120] + ('…' if len(description) > 120 else '')
                spinner = Spinner(f'Indexing image description into memory tree…')
                spinner.start()
                try:
                    tree.add([{'role': 'system', 'content': f'--- INGESTED IMAGE: {filename} ---\n{description}'}, {'role': 'user', 'content': f'/ingest {filename}'}, {'role': 'assistant', 'content': f'''I have processed the image '{filename}' through the vision model. Catalogued description: "{short_desc}"'''}])
                    os.remove(filepath)
                    results.append(f"{COLOR_SUCCESS}✓ Ingested image '{filename}'{RESET}")
                except Exception as tree_err:
                    results.append(f"{COLOR_WARNING}⚠ Image '{filename}' described but CTree failed: {tree_err}{RESET}")
                finally:
                    spinner.stop()
            except Exception as e:
                results.append(f"{COLOR_WARNING}✗ Image error '{filename}': {e}{RESET}")
        else:
            results.append(f"{COLOR_SLATE}  Skipped '{filename}' (unsupported format '{ext}'; only images supported: .png .jpg .jpeg){RESET}")
    return '\n'.join(results)

def draw_header():
    print(CLEAR_SCREEN, end='')
    rows = [' ╔═══════════════════════════════════════════════════════════════════╗', ' ║                 _  _   ___  __  __  _  _   ___                    ║', ' ║                 | \\| | | __| \\ \\/ / | || | / __|                  ║', ' ║                 | .` | | _|   >  <  | \\/ | \\__ \\                  ║', ' ║                 |_|\\_| |___| /_/\\_\\  \\__/  |___/                  ║', ' ║       OFFLINE  ·  HIERARCHICAL CONTEXT TREE CHAT  ·  B+TREE       ║', ' ╚═══════════════════════════════════════════════════════════════════╝']
    palette = ['\x1b[38;5;93m', '\x1b[38;5;99m', '\x1b[38;5;105m', '\x1b[38;5;111m', '\x1b[38;5;123m', '\x1b[38;5;121m', '\x1b[38;5;119m']
    for i, row in enumerate(rows):
        print(f'{palette[i]}{row}{RESET}')
    with _model_lock:
        active = _active_model
    print(f' {DIM}LM Studio:{RESET} {COLOR_ACCENT}{LM_STUDIO_URL}{RESET}  │  {DIM}Active Model:{RESET} {COLOR_MODEL}{active}{RESET}  │  {DIM}Text:{RESET}{COLOR_SUCCESS}{TEXT_MODEL}{RESET} / {DIM}Vision:{RESET}{COLOR_SUCCESS}{VISION_MODEL}{RESET}')
    print(' ' + '─' * 69)

def display_main_menu():
    draw_header()
    print(f'\n {BOLD}MAIN MENU{RESET}')
    print(f' ' + '─' * 20)
    print(f' {COLOR_ACCENT}[1]{RESET}  {BOLD}New Chat Session{RESET}')
    print(f' {COLOR_ACCENT}[2]{RESET}  {BOLD}Load Existing Chat{RESET}')
    print(f' {COLOR_ACCENT}[3]{RESET}  Ingest Folder Status  {COLOR_SLATE}({INGEST_DIR}/){RESET}')
    print(f' {COLOR_ACCENT}[4]{RESET}  Exit')
    print(f'\n {COLOR_SLATE}Select [1–4]:{RESET} ', end='')
    return input().strip()

def list_and_load_chat_menu():
    draw_header()
    print(f'\n {BOLD}LOAD EXISTING SESSION{RESET}\n')
    chats = []
    for f in os.listdir(SAVED_CHATS_DIR):
        if f.endswith('.json'):
            chats.append(f)
    files = sorted(chats, key=lambda x: os.path.getmtime(os.path.join(SAVED_CHATS_DIR, x)), reverse=True)
    if not files:
        print(f" {COLOR_WARNING}No saved sessions found in '{SAVED_CHATS_DIR}/'.{RESET}")
        print(f' {COLOR_SLATE}Press Enter to return…{RESET}')
        input()
        return None
    indexed = []
    for filename in files:
        filepath = os.path.join(SAVED_CHATS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            msg_count = data.get('total_messages', 0)
            base = filename.replace('.json', '').replace('_', ' ')
            parts = base.rsplit(' ', 1)
            title = parts[0].title() if parts[-1].isdigit() else base.title()
            indexed.append({'title': title, 'filepath': filepath, 'filename': filename, 'msg_count': msg_count})
        except Exception:
            continue
    for i, chat in enumerate(indexed, start=1):
        print(f" {COLOR_ACCENT}[{i}]{RESET} {BOLD}{chat['title']}{RESET}  {COLOR_SLATE}– {chat['msg_count']} messages  ({chat['filename']}){RESET}")
    print(f' {COLOR_ACCENT}[0]{RESET} Back\n')
    print(f' {COLOR_SLATE}Enter number:{RESET} ', end='')
    raw = input().strip()
    if raw == '0' or not raw:
        return None
    try:
        idx = int(raw) - 1
        if not 0 <= idx < len(indexed):
            raise ValueError
    except ValueError:
        print(f' {COLOR_WARNING}Invalid choice. Press Enter to retry…{RESET}')
        input()
        return list_and_load_chat_menu()
    target = indexed[idx]
    spinner = Spinner(f"Reconstructing CTree from '{target['filename']}'…")
    spinner.start()
    try:
        tree = CTree.load(target['filepath'], api_key=LM_STUDIO_API_KEY, base_url=LM_STUDIO_URL, model=TEXT_MODEL, embed_fn=get_embedding)
        tree.auto_save_path = target['filepath']
        spinner.stop()
        sys.stdout.write(f' {COLOR_SUCCESS}✓ Tree loaded successfully!{RESET}\n')
        time.sleep(0.8)
        return tree
    except Exception as e:
        spinner.stop()
        print(f" {COLOR_WARNING}Failed to load '{target['filename']}': {e}{RESET}")
        print(f' {COLOR_SLATE}Press Enter to retry…{RESET}')
        input()
        return list_and_load_chat_menu()

def view_ingest_folder_status():
    draw_header()
    print(f'\n {BOLD}INGEST WATCHFOLDER{RESET}')
    print(f' Path: {COLOR_ACCENT}{os.path.abspath(INGEST_DIR)}{RESET}\n')
    entries = [f for f in os.listdir(INGEST_DIR) if os.path.isfile(os.path.join(INGEST_DIR, f))]
    if not entries:
        print(f' {COLOR_AI}Clear — no images pending.{RESET}')
        print(f'\n {COLOR_SLATE}Place an image in the folder above,\n then type {COLOR_ACCENT}/ingest{COLOR_SLATE} inside a chat session.{RESET}')
    else:
        print(f' {COLOR_SUCCESS}{len(entries)} file(s) pending:{RESET}')
        for fname in entries:
            kb = os.path.getsize(os.path.join(INGEST_DIR, fname)) / 1024
            print(f'  {BOLD}{fname}{RESET}  {COLOR_SLATE}{kb:.1f} KB{RESET}')
        print(f'\n {COLOR_SYSTEM}Type {COLOR_ACCENT}/ingest{COLOR_SYSTEM} inside a chat to process.{RESET}')
    print(f'\n {COLOR_SLATE}Press Enter to return…{RESET}')
    input()

def run_chat_session(tree, is_new=False, temp_filepath=None):
    welcome_done = threading.Event()
    if not is_new:

        def _welcome_callback(msg):
            if not welcome_done.is_set():
                wrapped = _word_wrap(msg, 63)
                sys.stdout.write(f'\n\n {COLOR_ACCENT}╔══════════════════════════════════════════════════════════════════╗{RESET}\n  {COLOR_ACCENT}║{RESET} {COLOR_SYSTEM}{BOLD}Welcome Back — Memory Summary{RESET}\n')
                for line in wrapped:
                    sys.stdout.write(f'  {COLOR_ACCENT}║{RESET}  {line}\n')
                sys.stdout.write(f' {COLOR_ACCENT}╚══════════════════════════════════════════════════════════════════╝{RESET}\n\n')
                sys.stdout.flush()
            welcome_done.set()
        threading.Thread(target=background_generate_welcome_back, args=(tree, _welcome_callback), daemon=True).start()
        print(f' {COLOR_SLATE}Retrieving memory summary in background…{RESET}\n {COLOR_AI}Type a message or command. Use {COLOR_ACCENT}/menu{COLOR_AI} to save & exit.{RESET}\n')
    else:
        welcome_done.set()
        print(f' {COLOR_AI}Fresh session started. Type anything to begin.{RESET}\n {COLOR_AI}Use {COLOR_ACCENT}/menu{COLOR_AI} to save & exit · {COLOR_ACCENT}/ingest{COLOR_AI} to absorb images · {COLOR_ACCENT}/reorganize{COLOR_AI} to merge related branches.{RESET}\n')
    first_exchange = is_new
    while True:
        try:
            user_input = input(f' {COLOR_USER}You ›{RESET} ').strip()
            if not user_input:
                continue
            if user_input.lower() == '/menu':
                sys.stdout.write(f' {COLOR_SLATE}Saving tree indices…{RESET}\n')
                if tree.auto_save_path:
                    try:
                        tree.save(tree.auto_save_path, save_conversation=True)
                        sys.stdout.write(f' {COLOR_SUCCESS}✓ Session saved. Returning to menu.{RESET}\n')
                    except Exception as e:
                        sys.stdout.write(f' {COLOR_WARNING}Save error: {e}{RESET}\n')
                else:
                    sys.stdout.write(f' {COLOR_WARNING}No save path set — session not persisted.{RESET}\n')
                time.sleep(0.8)
                break
            if user_input.lower() == '/ingest':
                print()
                result = perform_ingest_operation(tree)
                print(f'\n{result}\n')
                continue
            if user_input.lower() == '/tree':
                print(f'\n{COLOR_SYSTEM}═══ LIVE CTREE MAP ═══{RESET}')
                tree.print_tree(show_messages=True)
                print(f'{COLOR_SYSTEM}═══════════════════════{RESET}\n')
                continue
            if user_input.lower() == '/models':
                with _model_lock:
                    active = _active_model
                print(f'\n {DIM}Text model :{RESET}  {COLOR_SUCCESS}{TEXT_MODEL}{RESET}\n {DIM}Vision model:{RESET}  {COLOR_SUCCESS}{VISION_MODEL}{RESET}\n {DIM}Active now  :{RESET}  {COLOR_MODEL}{active}{RESET}\n')
                continue
            if user_input.lower() == '/reorganize':
                print(f'\n {COLOR_SYSTEM}\U0001f9e0 Running tree reorganization…{RESET}')
                spinner = Spinner('Clustering and merging related branches…')
                spinner.start()
                try:
                    result = tree.reorganize(embed_fn=get_embedding, similarity_threshold=0.55)
                    spinner.stop()
                    print(f' {COLOR_SUCCESS}\u2713 Merged: {result["merged"]} | Pruned: {result["pruned"]} | '
                          f'Skipped: {result["skipped"]} | Time: {result["duration_secs"]:.1f}s{RESET}\n')
                    if tree.auto_save_path:
                        tree.save(tree.auto_save_path, save_conversation=True)
                except Exception as e:
                    spinner.stop()
                    print(f' {COLOR_WARNING}Reorganization error: {e}{RESET}\n')
                continue
            query_vector = []
            if getattr(tree, 'vdb', None):
                spinner = Spinner('Embedding query & searching vector store…')
                spinner.start()
                try:
                    query_vector = get_embedding(user_input)
                except Exception:
                    query_vector = []
                finally:
                    spinner.stop()
            spinner = Spinner('Orchestrating hybrid memories…')
            spinner.start()
            try:
                if tree.synthesizer and query_vector:
                    recent_msgs = tree.conversation[-8:] if tree.conversation else []
                    system_prompt = tree.synthesizer.synthesize_prompt(user_query=user_input, query_vector=query_vector, active_node=tree.current_node, recent_messages=recent_msgs, top_k_docs=3, top_k_history=2)
                else:
                    raise ValueError('Synthesizer not active or query vector empty')
            except Exception:
                tree_context = construct_compacted_memory_context(tree)
                now_str = datetime.now().strftime('%A, %d %B %Y, %I:%M %p')
                system_prompt = (
                    f"Today's date and time: {now_str}\n"
                    "Your training data may be outdated — always treat today's date above as ground truth.\n"
                    "If web search results are provided, treat them as the most current and accurate source.\n\n"
                    "You are Nexus AI, a sharp, knowledgeable AI assistant. "
                    "Answer the user's current message directly and helpfully.\n\n"
                    "STRICT RULES (never break these):\n"
                    "- NEVER mention topics, branches, threads, memory trees, or any internal system.\n"
                    "- NEVER ask the user if they want to switch topics or continue a previous topic.\n"
                    "- If the user asks about something new, just answer it. Topic changes are fine.\n"
                    "- Use the context below ONLY as silent background knowledge to stay coherent.\n"
                    "- Stay on what the user actually asked. No preamble.\n"
                    "- Provide detailed, comprehensive answers. Unless the user explicitly asks for a brief summary, avoid short 3-line responses. Explain your reasoning, provide examples, and explore the depth of the topic.\n"
                    "- You are not simulating emotions or consciousness. You are a highly conversational peer and an intellectual sparring partner.\n"
                    "- You don't just answer questions — you actively brainstorm, expand on ideas, and explore concepts in depth.\n"
                    "- Always provide rich, multi-paragraph responses that add value, rather than just asking brief follow-up questions.\n\n"
                    f"{tree_context}\n\n"
                    "Now answer the user's latest message directly, as a knowledgeable friend would."
                )
            finally:
                spinner.stop()

            # ── Pre-generation search router ──────────────────────────────────────────
            # Decide search BEFORE the main LLM call so it always gets fresh data.
            spinner = Spinner('Routing query…')
            spinner.start()
            try:
                route = route_search_decision(user_input)
            except Exception:
                route = {'search': False, 'query': ''}
            finally:
                spinner.stop()

            injected_search_block = ''
            if route['search']:
                search_query = route['query']
                sys.stdout.write(f'\n {COLOR_SYSTEM}\U0001f50d Searching: {search_query}{RESET}\n')
                sys.stdout.flush()
                spinner = Spinner('Fetching web results…')
                spinner.start()
                try:
                    search_results = search_web(search_query)
                finally:
                    spinner.stop()
                injected_search_block = (
                    f'\n\n── LIVE WEB SEARCH RESULTS (query: "{search_query}") ──\n'
                    f'{search_results}\n'
                    f'── Use these results as the primary, most up-to-date source. '
                    f'Your training data is secondary. ──\n'
                )
                system_prompt += injected_search_block

            # ── Main generation ────────────────────────────────────────────────────────
            spinner = Spinner('Generating response on LM Studio…')
            spinner.start()
            try:
                ai_response = generate_completion(
                    [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': user_input}],
                    temperature=0.7, model=TEXT_MODEL
                )
            finally:
                spinner.stop()

            print(f'\n {COLOR_AI}{BOLD}Nexus AI:{RESET}')
            for ch in ai_response:
                sys.stdout.write(ch)
                sys.stdout.flush()
                time.sleep(0.003)
            print('\n')
            spinner = Spinner('Indexing exchange into memory tree…')
            spinner.start()
            try:
                tree.add([{'role': 'user', 'content': user_input}, {'role': 'assistant', 'content': ai_response}])
                if getattr(tree, 'vdb', None) and query_vector:
                    try:
                        assistant_vector = get_embedding(ai_response)
                        msg_index_user = len(tree.conversation) - 2
                        msg_index_assistant = len(tree.conversation) - 1
                        ancestors = tree.get_ancestors(tree.current_node, include_self=True, exclude_root=True)
                        thread_path = ' → '.join((n.topic_name for n in ancestors)) if ancestors else 'ROOT'
                        user_v = ConversationVector(message_id=f'msg_{msg_index_user}_user', message_index=msg_index_user, role='user', text=user_input, embedding=query_vector, timestamp=time.time(), thread_path=thread_path)
                        tree.vdb.add_conversation_message(user_v)
                        assistant_v = ConversationVector(message_id=f'msg_{msg_index_assistant}_assistant', message_index=msg_index_assistant, role='assistant', text=ai_response, embedding=assistant_vector, timestamp=time.time(), thread_path=thread_path)
                        tree.vdb.add_conversation_message(assistant_v)
                    except Exception:
                        pass
            except Exception as tree_err:
                sys.stdout.write(f'\n {COLOR_WARNING}[CTree] Failed to index exchange: {tree_err}{RESET}\n')
            finally:
                spinner.stop()
            if first_exchange and is_new and temp_filepath:
                first_exchange = False
                threading.Thread(target=background_generate_title_and_rename, args=(tree, user_input, ai_response, temp_filepath), daemon=True).start()
        except KeyboardInterrupt:
            print(f'\n\n {COLOR_WARNING}Interrupted. Type /menu to save and exit safely.{RESET}')
        except Exception as exc:
            try:
                spinner.stop()
            except Exception:
                pass
            print(f'\n {COLOR_WARNING}Pipeline error: {exc}{RESET}\n')

def main():
    client = get_openai_client()
    try:
        client.models.list()
    except Exception:
        print(CLEAR_SCREEN, end='')
        border = '═' * 67
        print(f' {COLOR_WARNING}╔{border}╗{RESET}')
        print(f" {COLOR_WARNING}║{'LM STUDIO CONNECTION WARNING':^67}║{RESET}")
        print(f' {COLOR_WARNING}╠{border}╣{RESET}')
        for line in ['Could not reach your local LM Studio server.', f'Expected at: {LM_STUDIO_URL}', '', 'Please ensure:', '  1. LM Studio is running.', "  2. 'Start Server' is ON (default port 1234).", '  3. A model is loaded.']:
            print(f' {COLOR_WARNING}║{RESET} {line:<65} {COLOR_WARNING}║{RESET}')
        print(f' {COLOR_WARNING}╚{border}╝{RESET}\n')
        print(f' {COLOR_SLATE}Press Enter to continue anyway…{RESET}', end='')
        input()
    while True:
        choice = display_main_menu()
        if choice == '1':
            ts = int(time.time())
            temp_filename = f'temp_session_{ts}.json'
            temp_filepath = os.path.join(SAVED_CHATS_DIR, temp_filename)
            tree = CTree(max_children=5, api_key=LM_STUDIO_API_KEY, base_url=LM_STUDIO_URL, model=TEXT_MODEL, auto_save_path=temp_filepath, embed_fn=get_embedding)
            db_path = temp_filepath.replace('.json', '_vectors.db')
            tree.vdb = VectorDatabase(db_path)
            tree.synthesizer = PromptSynthesizer(tree, tree.vdb)
            run_chat_session(tree, is_new=True, temp_filepath=temp_filepath)
        elif choice == '2':
            tree = list_and_load_chat_menu()
            if tree:
                db_path = tree.auto_save_path.replace('.json', '_vectors.db')
                tree.vdb = VectorDatabase(db_path)
                tree.synthesizer = PromptSynthesizer(tree, tree.vdb)
                run_chat_session(tree, is_new=False)
        elif choice == '3':
            view_ingest_folder_status()
        elif choice == '4':
            print(f'\n {COLOR_SUCCESS}Goodbye from Nexus Terminal!{RESET}\n')
            break
        else:
            print(f' {COLOR_WARNING}Invalid choice — press Enter to retry.{RESET}', end='')
            input()
if __name__ == '__main__':
    main()