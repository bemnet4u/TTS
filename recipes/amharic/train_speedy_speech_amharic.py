import os

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.speedy_speech_config import SpeedySpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    name="myvoice", meta_file_train="metadata.csv", path=os.path.join(output_path, "/content/drive/MyDrive/VoiceCloning/datasets/Speaker2_Coqui")
)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    power=1.4,
    preemphasis=0.5,
    ref_level_db=10,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)

config = SpeedySpeechConfig(
    run_name="speedy_speech_amharic",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    log_model_step=500,
    save_step=2500,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    phoneme_language="am-ET",
    use_espeak_phonemes=False,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    sort_by_audio_len=True,
    max_seq_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
    characters={
        "pad": "_",
        "eos": "&",
        "bos": "*",
        "characters": "ሀሁሂሃሄህሆሇለሉሊላሌልሎሏሐሑሒሓሔሕሖሗመሙሚማሜምሞሟሠሡሢሣሤሥሦሧረሩሪራሬርሮሯሰሱሲሳሴስሶሷሸሹሺሻሼሽሾሿቀቁቂቃቄቅቆቇቈቊቋቌቍቐቑቒቓቔቕቖቘቚቛቜቝበቡቢባቤብቦቧቨቩቪቫቬቭቮቯተቱቲታቴትቶቷቸቹቺቻቼችቾቿኀኁኂኃኄኅኆኇኈኊኋኌኍነኑኒናኔንኖኗኘኙኚኛኜኝኞኟአኡኢኣኤእኦኧከኩኪካኬክኮኯኰኲኳኴኵኸኹኺኻኼኽኾዀዂዃዄዅወዉዊዋዌውዎዏዐዑዒዓዔዕዖዘዙዚዛዜዝዞዟዠዡዢዣዤዥዦዧየዩዪያዬይዮዯደዱዲዳዴድዶዷዸዹዺዻዼዽዾዿጀጁጂጃጄጅጆጇገጉጊጋጌግጎጏጐጒጓጔጕጘጙጚጛጜጝጞጟጠጡጢጣጤጥጦጧጨጩጪጫጬጭጮጯጰጱጲጳጴጵጶጷጸጹጺጻጼጽጾጿፀፁፂፃፄፅፆፇፈፉፊፋፌፍፎፏፐፑፒፓፔፕፖፗፘፙ",
        "punctuations": "።!'(),.:;?፠፡፣፤፥፦፧፨ ",
        "phonemes": None,
        "unique": True,
    },
    test_sentences=["የአሕዛብ አስተማሪ ለመሆን ተሾምሁእውነት እናገራለሁ አልዋሽም።", 
                    "እግዚአብሔርን እንፈራለን ለሚሉት ሴቶች እንደሚገባ መልካም በማድረግ እንጂ።", 
                    "ይህን ለማግኘት እንደክማለንና ስለዚህም እንሰደባለንይህም ሰውን ሁሉ ይልቁንም", 
                    "በትንቢት ከሽማግሌዎች እጅ መጫን ጋር የተሰጠህን በአንተ ያለውን የጸጋ ስጦታ ቸል አትበል።"
    ]
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init model
model = ForwardTTS(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()
