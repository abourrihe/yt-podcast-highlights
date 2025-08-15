# scripts/process_video.py
import os, sys, argparse, json, subprocess
from typing import List

# Allow running as plain script (without -m)
sys.path.append(os.path.dirname(__file__))

import pysrt
try:
    from .utils import Segment
except ImportError:
    from utils import Segment

def run(cmd: list[str]):
    print('+', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)

# ---------------------------
# Transcription
# ---------------------------
def transcribe_whisper(
    audio_path: str,
    lang_hint: str = "ar",
) -> List[Segment]:
    """
    Transcribes audio using faster-whisper with env-tunable settings.
    Env vars:
      WHISPER_MODEL  (default: 'medium')  e.g., 'small','medium','large-v3'
      WHISPER_BEAM   (default: '1')
      WHISPER_VAD    (default: '1')
      WHISPER_CHUNK  (default: '30')   seconds
      WHISPER_COMPUTE(default: 'int8') e.g., 'int8','int8_float16','float16','float32'
      WHISPER_DEVICE (default: 'cpu')  e.g., 'cpu','cuda'
    """
    from faster_whisper import WhisperModel

    model_name   = os.getenv("WHISPER_MODEL", "medium")
    beam_size    = int(os.getenv("WHISPER_BEAM", "1"))
    vad_filter   = os.getenv("WHISPER_VAD", "1") == "1"
    chunk_size_s = int(os.getenv("WHISPER_CHUNK", "30"))
    compute_type = os.getenv("WHISPER_COMPUTE", "int8")
    device       = os.getenv("WHISPER_DEVICE", "cpu")

    print(f"[whisper] model={model_name} device={device} compute={compute_type} "
          f"beam={beam_size} vad={vad_filter} chunk={chunk_size_s}s", flush=True)

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        audio_path,
        language=lang_hint,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=250),
        beam_size=beam_size,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
        chunk_size=chunk_size_s,
    )

    out: List[Segment] = []
    for seg in segments:
        out.append(Segment(
            start=float(seg.start),
            end=float(seg.end),
            text=(seg.text or "").strip(),
            avg_logprob=float(getattr(seg, "avg_logprob", 0.0)) if getattr(seg, "avg_logprob", None) is not None else None
        ))
    return out

def write_transcript_json_srt(segments: List[Segment], out_json: str, out_srt: str):
    # JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in segments], f, ensure_ascii=False, indent=2)

    # SRT
    subs = pysrt.SubRipFile()
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return pysrt.SubRipTime(hours=h, minutes=m, seconds=s, milliseconds=ms)

    for i, s in enumerate(segments, 1):
        subs.append(pysrt.SubRipItem(index=i, start=fmt(s.start), end=fmt(s.end), text=s.text))
    subs.save(out_srt, encoding="utf-8")

# ---------------------------
# Export clip with/without burn
# ---------------------------
def load_and_shift_srt_for_clip(srt_path: str, start_sec: float, end_sec: float) -> pysrt.SubRipFile:
    subs = pysrt.open(srt_path, encoding="utf-8")
    clip = pysrt.SubRipFile()
    for it in subs:
        st = it.start.ordinal / 1000.0
        en = it.end.ordinal / 1000.0
        if en <= start_sec or st >= end_sec:
            continue
        # overlap
        st_clip = max(0.0, st - start_sec)
        en_clip = max(0.01, en - start_sec)
        new_it = pysrt.SubRipItem(
            index=len(clip) + 1,
            start=pysrt.SubRipTime(milliseconds=int(st_clip * 1000)),
            end=pysrt.SubRipTime(milliseconds=int(en_clip * 1000)),
            text=it.text
        )
        clip.append(new_it)
    return clip

def export_clip_raw(video_path: str, start: float, end: float, out_path: str):
    dur = max(0.01, end - start)
    run(['ffmpeg','-y','-ss', f"{start:.2f}", '-i', video_path, '-t', f"{dur:.2f}",
         '-c','copy', out_path])

def burn_subs(in_path: str, out_path: str, srt_path: str, font_path: str,
              logo_path: str | None, end_slate: str | None):
    # Build filter chain
    vf = [f"subtitles='{srt_path}':fontsdir='{os.path.dirname(font_path)}'"]
    vf_filt = ','.join(vf)

    run(['ffmpeg','-y','-i', in_path, '-vf', vf_filt,
         '-c:v','libx264','-preset','veryfast','-crf','23','-c:a','aac','-b:a','128k', out_path])

    if end_slate and os.path.exists(end_slate):
        tmp = out_path.replace('.mp4','_with_end.mp4')
        run(['ffmpeg','-y','-i', out_path, '-i', end_slate, '-filter_complex',
             '[0:v][1:v]concat=n=2:v=1:a=0', '-c:v','libx264','-preset','veryfast','-crf','23', tmp])
        os.replace(tmp, out_path)

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['transcribe','export'])

    # transcribe
    ap.add_argument('--audio', help='input audio wav (16k mono recommended)')
    ap.add_argument('--out_dir', default='.', help='output dir for transcript files')
    ap.add_argument('--lang', default='ar')

    # export
    ap.add_argument('--video', help='input video mp4')
    ap.add_argument('--srt', help='full transcript srt')
    ap.add_argument('--start', type=float, help='clip start (seconds)')
    ap.add_argument('--end', type=float, help='clip end (seconds)')
    ap.add_argument('--out', help='output clip path (mp4)')
    ap.add_argument('--font', default='assets/font.ttf')
    ap.add_argument('--logo', default=None)
    ap.add_argument('--end_slate', default=None)
    ap.add_argument('--burn', action='store_true', help='burn subtitles into video')

    args = ap.parse_args()

    if args.stage == 'transcribe':
        if not args.audio:
            ap.error("--audio is required for stage=transcribe")
        os.makedirs(args.out_dir, exist_ok=True)
        segs = transcribe_whisper(args.audio, lang_hint=args.lang)
        out_json = os.path.join(args.out_dir, "transcript.json")
        out_srt  = os.path.join(args.out_dir, "transcript.srt")
        write_transcript_json_srt(segs, out_json, out_srt)
        print(f"Wrote {out_json} and {out_srt}")

    elif args.stage == 'export':
        for req in ('video','srt','start','end','out'):
            if getattr(args, req) is None:
                ap.error(f"--{req} is required for stage=export")

        # 1) cut raw clip
        tmp_raw = args.out.replace('.mp4','_raw.mp4')
        export_clip_raw(args.video, float(args.start), float(args.end), tmp_raw)

        # 2) subtitles for the clip
        sub_clip = load_and_shift_srt_for_clip(args.srt, float(args.start), float(args.end))
        tmp_srt = args.out.replace('.mp4','_clip.srt')
        sub_clip.save(tmp_srt, encoding='utf-8')

        # 3) optional burn
        if args.burn or args.font:
            font_path = args.font or 'assets/font.ttf'
            burn_subs(tmp_raw, args.out, tmp_srt, font_path, args.logo, args.end_slate)
        else:
            # no burn: just rename the raw cut to final
            os.replace(tmp_raw, args.out)

        print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()
