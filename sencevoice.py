from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "/data/guyf/.cache/modelscope/hub/iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    vad_model="/data/guyf/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
#text = rich_transcription_postprocess(res[0]["text"])
print(res)
'''
# generate train.jsonl and val.jsonl from wav.scp and text.txt
sensevoice2jsonl \
++scp_file_list='["/data/guyf/funasr/FunASR_sv/data/list/wav_tra.scp", "/data/guyf/funasr/FunASR_sv/data/list/text_tra"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="/data/guyf/funasr/FunASR_sv/data/train.jsonl" \
++model_dir='iic/SenseVoiceSmall'
'''