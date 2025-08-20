# -*- coding: utf-8 -*-
"""AudioToText : traitement séquentiel des fichiers .m4v sur GPU,
sortie TXT uniquement + barre de progression détaillée par fichier + transcription en français forcée + extraction audio"""
import os
import subprocess
import tempfile

import torch
import whisper
from tqdm import tqdm
from pydub import AudioSegment

# Configuration (modifiez si nécessaire)
INPUT_DIR = "."                    # Répertoire contenant vos fichiers .m4v
MODEL_NAME = "large-v3"               # Modèle Whisper à utiliser
FORCED_LANGUAGE = "fr"             # Code ISO pour forcer la transcription en français (None pour auto-détect)
OUTPUT_DIR = "audio_transcription"   # Répertoire de sortie pour les fichiers .txt
VIDEO_EXTENSIONS = {'.m4v', '.mp4', '.mov', '.avi', '.mkv'}
CHUNK_MS = 10000                    # Durée d'un chunk en milliseconds (10s)


def extract_audio(video_path: str) -> str:
    """Extrait l'audio d'un fichier vidéo et renvoie le chemin du fichier WAV temporaire."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-ac', '1', '-ar', '16000', tmp_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp_wav


def transcribe_file(audio_path: str, model) -> None:
    """Transcrit un fichier audio/vidéo en découpant en segments et affiche la progression."""
    # Extraction audio
    ext = os.path.splitext(audio_path)[1].lower()
    cleanup = False
    path_to_transcribe = audio_path
    if ext in VIDEO_EXTENSIONS:
        try:
            path_to_transcribe = extract_audio(audio_path)
            cleanup = True
        except subprocess.CalledProcessError:
            print(f"[Erreur] impossible d'extraire l'audio de {audio_path}")
            return

    # Chargement du segment audio complet
    audio_seg = AudioSegment.from_file(path_to_transcribe)
    total_ms = len(audio_seg)
    # Découpage en chunks
    chunks = [audio_seg[i:i + CHUNK_MS] for i in range(0, total_ms, CHUNK_MS)]

    # Transcription segment par segment avec barre interne
    texts = []
    for chunk in tqdm(chunks, desc=f"Transcrire {os.path.basename(audio_path)}", leave=False):
        # Sauvegarde du chunk temporaire
        tmp_chunk = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        chunk.export(tmp_chunk, format='wav')
        # Options Whisper
        options = {}
        if FORCED_LANGUAGE:
            options['language'] = FORCED_LANGUAGE
        # Transcrire
        result = model.transcribe(tmp_chunk, **options)
        texts.append(result.get('text', '').strip())
        # Nettoyage
        os.remove(tmp_chunk)

    full_text = '\n'.join(texts)

    # Nettoyage du fichier audio complet
    if cleanup and os.path.exists(path_to_transcribe):
        os.remove(path_to_transcribe)

    # Enregistrement du résultat
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement du modèle {MODEL_NAME} sur {device}")
    model = whisper.load_model(MODEL_NAME, device=device)

    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
             if os.path.isfile(os.path.join(INPUT_DIR, f)) and f.lower().endswith(tuple(VIDEO_EXTENSIONS))]
    if not files:
        print(f"Aucun fichier vidéo ({', '.join(VIDEO_EXTENSIONS)}) trouvé dans: {INPUT_DIR}")
    else:
        for audio_path in files:
            try:
                transcribe_file(audio_path, model)
            except Exception as e:
                print(f"[Erreur] {audio_path} : {e}")
        print("Toutes les transcriptions sont terminées.")