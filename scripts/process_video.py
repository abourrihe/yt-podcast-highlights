import argparse, json, os, subprocess
from typing import List
from faster_whisper import WhisperModel
import pysrt
from utils import Segment
from scoring import choose_clip

def run(cmd: list[str]):
    print('+', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)

def transcribe(path: str, lang_hint: str = 'ar') -> List[Segment]:
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(path, language=lang_hint, vad_filter=True)
    out = []
    for seg in segments:
        out.append(Segment(float(seg.start), float(seg.end), seg.text.strip(), seg.avg_logprob))
    return out

def write_srt(segments: List[Segment], srt_path: str):
    subs = pysrt.SubRipFile()
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600)//60); s = int(t%60); ms = int((t - int(t))*1000)
        return pysrt.SubRipTime(hours=h, minutes=m, seconds=s, milliseconds=ms)
    for i, s in enumerate(segments, 1):
        subs.append(pysrt.SubRipItem(index=i, start=fmt(s.start), end=fmt(s.end), text=s.text))
    subs.save(srt_path, encoding='utf-8')

def burn_subs(in_path: str, out_path: str, srt_path: str, font_path: str,
              logo_path: str | None, end_slate: str | None):
    vf = [f"subtitles='{srt_path}':fontsdir='{os.path.dirname(font_path)}'"]
    if logo_path and os.path.exists(logo_path):
        vf.append("movie='{}',scale=iw*0.12:-1 [logo]; [in][logo] overlay=W-w-30:30 [out]".format(logo_path))
    vf_filt = ','.join(vf)
    run(['ffmpeg','-y','-i', in_path, '-vf', vf_filt,
         '-c:v','libx264','-preset','veryfast','-crf','23','-c:a','aac','-b:a','128k', out_path])
    if end_slate and os.path.exists(end_slate):
        tmp = out_path.replace('.mp4','_with_end.mp4')
        run(['ffmpeg','-y','-i', out_path, '-i', end_slate, '-filter_complex',
             '[0:v][1:v]concat=n=2:v=1:a=0', '-c:v','libx264','-preset','veryfast','-crf','23', tmp])
        os.replace(tmp, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--min-seconds', type=int, default=30)
    ap.add_argument('--max-seconds', type=int, default=60)
    ap.add_argument('--lang', default='ar')
    ap.add_argument('--keywords-prefer', default='')
    ap.add_argument('--keywords-avoid', default='')
    ap.add_argument('--font', required=True)
    ap.add_argument('--logo', default='assets/logo.png')
    ap.add_argument('--end-slate', default='assets/end_slate.png')
    args = ap.parse_args()

    segs = transcribe(args.input, lang_hint=args.lang)
    clip = choose_clip(segs, args.min_seconds, args.max_seconds,
                       [k.strip() for k in args.keywords_prefer.split(',') if k.strip()],
                       [k.strip() for k in args.keywords_avoid.split(',') if k.strip()])
    if not clip:
        raise SystemExit('no suitable clip found')

    out_clip = 'clip.mp4'
    run(['ffmpeg','-y','-ss', f"{clip.start}", '-i', args.input, '-to', f"{clip.end-clip.start}", '-c','copy', out_clip])

    sub_segments = [s for s in segs if s.start < clip.end and s.end > clip.start]
    for s in sub_segments:
        s.start = max(0.0, s.start - clip.start)
        s.end = max(0.01, s.end - clip.start)
    srt_path = 'clip.srt'
    write_srt(sub_segments, srt_path)

    outfile = 'clip_burned.mp4'
    burn_subs(out_clip, outfile, srt_path, args.font, args.logo, args.end_slate)

    meta = {
        'outfile': outfile,
        'start': round(clip.start, 2),
        'end': round(clip.end, 2),
        'text': clip.text,
        'channel': os.getenv('GITHUB_REPOSITORY', '').split('/')[-1]
    }
    with open('artifact.json','w',encoding='utf-8') as f:
        import json; json.dump(meta, f, ensure_ascii=False)

if __name__ == '__main__':
    main()
