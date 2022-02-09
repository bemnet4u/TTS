#!flask/bin/python
import argparse
from array import array
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Union
from ast import literal_eval
from uuid import uuid4
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, abort, jsonify, render_template, request, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.getLevelName('INFO'))
log_formatter = logging.Formatter("%(asctime)s tid=%(thread)d tname=[%(threadName)s] [%(levelname)s] %(name)s: %(message)s  ")

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
jobs = dict()

def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    parser.add_argument("--rate_limit", type=str, default="", help='Rate limit. Eg. ["200 per day", "50 per hour"].')
    parser.add_argument("--executors", type=str, default=5, help='Number of executors')
    return parser


# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custome model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

executor = ThreadPoolExecutor(args.executors)

# load models
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

use_multi_speaker = hasattr(synthesizer.tts_model, "num_speakers") and synthesizer.tts_model.num_speakers > 1
speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)
# TODO: set this from SpeakerManager
use_gst = synthesizer.tts_config.get("use_gst", False)
app = Flask(__name__)

if args.rate_limit is not None and args.rate_limit != "":
    limit = literal_eval(args.rate_limit)
    logger.info("Using rate limit: {}".format(limit))
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=limit
    )

def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=use_multi_speaker,
        speaker_ids=speaker_manager.speaker_ids if speaker_manager is not None else None,
        use_gst=use_gst,
    )


@app.route("/details")
def details():
    model_config = load_config(config_path)
    if vocoder_config_path is not None and os.path.isfile(vocoder_config_path):
        vocoder_config = load_config(vocoder_config_path)
    else:
        vocoder_config = None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


@app.route("/api/tts", methods=["GET"])
def tts():
    text = request.args.get("text")
    speaker_idx = request.args.get("speaker_id", "")
    style_wav = request.args.get("style_wav", "")

    style_wav = style_wav_uri_to_dict(style_wav)
    logger.info(" > Model input: {}".format(text))
    wavs = synthesizer.tts(text, speaker_name=speaker_idx, style_wav=style_wav)
    out = io.BytesIO()
    synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


@app.route("/api/tts_async", methods=["GET"])
def tts_async():
    text = unquote(request.args.get("text"))
    speaker_idx = unquote(request.args.get("speaker_id", ""))
    style_wav = unquote(request.args.get("style_wav", ""))
    id = str(uuid4())
    logger.info("Submitting request with id %s, text: %s", id, text)
    executor.submit(_tts_async, id, text, speaker_idx, style_wav)
    jobs[id] = {"status": "submitted"}
    return jsonify({"id": id, "status": "submitted"})


def _tts_async(id, text, speaker_idx, style_wav):
    try:
        jobs[id] = {"status": "running"}
        style_wav = style_wav_uri_to_dict(style_wav)
        logger.info("Running model with id: %s input: %s", id, text)
        wavs = synthesizer.tts(text, speaker_name=speaker_idx, style_wav=style_wav)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
        jobs[id] = {"status": "finished", "audio": out}
        logger.info("Finished synthesising id: %s input: %s", id, text)
    except Exception as ex:
        logger.error(ex)
        jobs[id] = {"status": "error", "message": str(ex)}

@app.route("/api/tts_async/status", methods=["GET"])
def tts_async_status():
    id = unquote(request.args.get("id"))
    if id in jobs:
        status = jobs[id]["status"]
        logger.info("Querying model status id: %s status: %s", id, status)
        return jsonify({"status": status})
    else:
        abort(404)

@app.route("/api/tts_async/audio", methods=["GET"])
def tts_async_audio():
    id = unquote(request.args.get("id"))
    logger.info("tts_async_audio called with id: %s", id)
    if id in jobs and jobs[id]["status"] == 'finished' and jobs[id]["audio"] is not None:
        logger.info("Downloading audio with id: %s", id)
        out = jobs.pop(id)
        return send_file(out["audio"], mimetype="audio/wav")
    else:
        abort(404)

def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()
